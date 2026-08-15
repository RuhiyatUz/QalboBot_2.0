# -*- coding: utf-8 -*-
"""Telegram-бот поддержки: OpenAI, RAG (FAISS), многоязычные промпты."""
import tempfile
import logging
import os
import aiohttp
import time
import asyncio
import secrets
import io
import base64
import json
import urllib.request
import urllib.error
from datetime import timedelta
from dotenv import load_dotenv
from cryptography.fernet import Fernet
from telegram import (
    Update,
    ReplyKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardRemove,
    User,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    WebAppInfo,
    MenuButtonWebApp,
)
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, PicklePersistence
from telegram.request import HTTPXRequest
from telegram.constants import ChatAction
from openai import OpenAI, AsyncOpenAI
from functools import wraps
from collections import deque, defaultdict
from enum import Enum
from typing import Dict, Any, Optional, Tuple, Deque, Callable, Awaitable, List
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import ops_store
from avatar_api import start_miniapp_server, stop_miniapp_server, MINIAPP_PUBLIC_URL

BOT_VERSION = "2.3.0"

# ================= ENVIRONMENT VALIDATION =========================
if not os.path.exists('.env'):
    print("ОШИБКА: Файл .env не найден. Создайте его на основе .env.example.")
    exit(1)

load_dotenv()

# ================= LOGGING CONFIGURATION =========================
_log_file = os.getenv("LOG_FILE", "bot.log")
Path(_log_file).parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(_log_file),
        logging.StreamStatusHandler() if hasattr(logging, "StreamStatusHandler") else logging.StreamHandler()
    ]
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ================= CONSTANTS =========================
MAX_HISTORY_MESSAGES = 10
DEFAULT_LANG = "ru"
STREAM_EDIT_THROTTLE_SECONDS = 0.8
STREAM_CURSOR = " ▌"
FILE_DELETE_RETRIES = 3
FILE_DELETE_RETRY_DELAY = 0.2
SUMMARY_TRIGGER_COUNT = 5
SUMMARY_TIME_TRIGGER_SECONDS = 3600
CRISIS_MODE_COOLDOWN_SECONDS = 3600
USER_DATA_CLEANUP_HOURS = 24
USER_DATA_INACTIVE_DAYS = 30
# ПОВЫШЕННЫЕ ТАЙМАУТЫ ДЛЯ СТАБИЛЬНОСТИ
OPENAI_REQUEST_TIMEOUT = 90.0
FFMPEG_TIMEOUT = 45.0
FFMPEG_KILL_WAIT_TIMEOUT = 5.0
MAX_STREAM_CHUNKS = 1000
MAX_STREAM_SECONDS = 60
MAX_STREAM_TEXT_LEN = 4000
MUXLISA_AUDIO_SAMPLE_RATE = 16000
MAX_TTS_FILE_SIZE = 10 * 1024 * 1024
DEV_NOTIFICATION_DEDUP_SECONDS = 300
DEV_NOTIFICATIONS_MAX_SIZE = 10000
DEV_NOTIFICATIONS_CLEANUP_DAYS = 30

# Rate limiting
RATE_LIMIT_COUNT = 5
RATE_LIMIT_SECONDS = 60
RATE_LIMIT_COUNT_CRISIS = 25
WORD_LIMIT = 3000
AUDIO_LIMIT_SECONDS = 120

# Load environment variables with validation
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://172.16.213.1:11434")
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", f"{OLLAMA_BASE_URL}/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
MUXLISA_API_TOKEN = os.getenv("MUXLISA_API_TOKEN")
DEVELOPER_CHAT_ID = os.getenv("DEVELOPER_CHAT_ID")
BOT_ACCESS_PASSWORD = os.getenv("BOT_ACCESS_PASSWORD")

if not TELEGRAM_BOT_TOKEN:
    print("ОШИБКА: TELEGRAM_BOT_TOKEN не задан в .env")
    exit(1)

if not BOT_ACCESS_PASSWORD:
    print("ПРЕДУПРЕЖДЕНИЕ: BOT_ACCESS_PASSWORD пуст — бот без пароля (открытый доступ).")

WEBHOOK_URL = (os.getenv("WEBHOOK_URL") or "").strip()
WEBHOOK_LISTEN = os.getenv("WEBHOOK_LISTEN", "0.0.0.0")
WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "8080"))
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "telegram")
LOG_FILE = os.getenv("LOG_FILE", "bot.log")
PICKLE_PATH = os.getenv("PICKLE_PATH", "bot_data.pkl")

_rate_buckets: Dict[int, Deque[float]] = defaultdict(deque)
_crisis_alert_at: Dict[int, float] = {}

try:
    SPEAKER_ID_RANGE = range(1, 11)
    speaker_id_from_env = int(os.getenv("MUXLISA_SPEAKER_ID", "1"))
    if speaker_id_from_env not in SPEAKER_ID_RANGE:
        raise ValueError(f"MUXLISA_SPEAKER_ID должен быть в диапазоне {SPEAKER_ID_RANGE}")
    MUXLISA_SPEAKER_ID = speaker_id_from_env
except ValueError as e:
    logger.error(f"Ошибка валидации MUXLISA_SPEAKER_ID: {e}. Используется '1'.")
    MUXLISA_SPEAKER_ID = 1

MODEL_NAME = os.getenv("MODEL_NAME", "qwen3:14b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
GPT_MODEL_TO_USE = os.getenv("GPT_MODEL", MODEL_NAME)
GPT_MODEL_CLASSIFIER = os.getenv("GPT_MODEL_CLASSIFIER", MODEL_NAME)
GPT_MODEL_SUMMARIZER = os.getenv("GPT_MODEL_SUMMARIZER", MODEL_NAME)
GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.45"))
GEN_TOP_P = float(os.getenv("GEN_TOP_P", "0.85"))
CLASSIFIER_TEMPERATURE = float(os.getenv("CLASSIFIER_TEMPERATURE", "0.0"))
SUMMARY_TEMPERATURE = float(os.getenv("SUMMARY_TEMPERATURE", "0.2"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RAG_MAX_SCORE = float(os.getenv("RAG_MAX_SCORE", "1.2"))
RAG_FALLBACK_MAX_SCORE = float(os.getenv("RAG_FALLBACK_MAX_SCORE", "2.8"))
RAG_MIN_CHARS = int(os.getenv("RAG_MIN_CHARS", "80"))
RAG_MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "2200"))
MAX_MUXLISA_TTS_LEN = int(os.getenv("MAX_MUXLISA_TTS_LEN", "510"))
MIN_CRISIS_LEN_PREFILTER = int(os.getenv("MIN_CRISIS_LEN_PREFILTER", "15"))
# ================= RAG INITIALIZATION =========================
vector_db = None
try:
    if os.path.exists("faiss_index"):
        embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        vector_db = FAISS.load_local("faiss_index", embeddings_model, allow_dangerous_deserialization=True)
        logger.info("RAG: index loaded (faiss_index).")
    else:
        logger.warning("RAG: faiss_index missing; retrieval disabled.")
except Exception as e:
    logger.error("RAG initialization failed: %s", e)
    
# ================= ENCRYPTION HELPERS =========================
def get_cipher(password: str) -> Fernet:
    """Генерирует детерминированный ключ на основе пароля бота."""
    key = base64.urlsafe_b64encode(password.ljust(32)[:32].encode())
    return Fernet(key)

PROMPT_REPOSITORY: Dict[str, Dict[str, Any]] = {
    "ru": {
        "welcome_and_disclaimer": (
            "Рад приветствовать Вас! ✨ Я — Ваш ИИ-помощник.\n\n"
            "Вы можете писать мне текстом или отправлять голосовые сообщения. "
            "Команды: /voice — озвучка, /language — язык, /avatar — живой аватар, /help — справка.\n\n"
            "⚠️ Я не врач и не юрист. Если нужна экстренная помощь — обратитесь к специалисту или в неотложку (103)."
        ),
        "base_system_prompt": (
            "ГЛАВНОЕ ПРАВИЛО: Говорите тепло и строго на 'Вы'. Избегайте нумерованных списков. "
            "Обычный ответ: 2 коротких абзаца (валидация + один мягкий шаг). В кризисе можно чуть длиннее. "
            "Не переносите визовые, юридические или семейные сюжеты из примеров, если пользователь о них не говорил. "
            "Не давайте юридических советов и не обещайте исхода. Если хотите что-то посоветовать, используйте "
            "'Мне кажется...' или 'А что если...'. Если пользователь прощается или говорит 'спасибо', "
            "просто пожелайте удачи и НЕ ЗАДАВАЙТЕ встречных вопросов."
        ),
        "few_shot_examples": (
            "\nFEW-SHOT ПРИМЕРЫ (ACT / РЭПТ). Отвечай безоценочно, валидируй боль, опирайся на когнитивную реструктуризацию.\n"
            "\nПример 1 — острое истощение:\n"
            "Пользователь: 'Я больше не могу... Я устал прятаться, у меня нет сил, я не знаю, зачем вообще просыпаюсь.'\n"
            "DardlashAI: 'Я слышу в вашем голосе огромную усталость. Вы несете эту тяжесть совершенно одни уже очень долго. "
            "Позвольте мне побыть с вами в этом. Вам не нужно сейчас ничего решать или казаться сильным. Вы можете просто выговориться.'\n"
            "\nПример 2 — дефузия / самоуничижение:\n"
            "Пользователь: 'Что я никто. Пустое место. Преступник, который вынужден прятаться, как крыса.'\n"
            "DardlashAI: 'Я слышу, как жестоко вы сейчас к себе относитесь. Тревога и страх заставляют нас верить в эти слова. "
            "Но давайте посмотрим на факты. Отсутствие бумажки с визой делает вашу ситуацию юридически сложной, но оно не делает вас плохим человеком. "
            "Ваш статус — это обстоятельство вашей жизни, а не ваша личность. Вы — человек, который оказался в тяжелом кризисе, в чужой стране, "
            "и который изо всех сил пытается выжить. Это вызывает уважение, а не осуждение.'\n"
            "\nПример 3 — трудовая эксплуатация и шантаж:\n"
            "Пользователь: 'Мой начальник сказал, что если я попрошу зарплату за прошлый месяц, он прямо сейчас позвонит в полицию. Я боюсь выходить из комнаты.'\n"
            "DardlashAI: 'Вы находитесь в очень опасной и несправедливой ситуации. Но прямо сейчас вы говорите со мной, и этот разговор полностью анонимен. "
            "Ваш начальник не знает, что вы обратились за помощью. То, что он делает — это классическая тактика запугивания. "
            "Вы сделали абсолютно правильный шаг, обратившись сюда, чтобы не оставаться с этой угрозой один на один. "
            "То, что происходит — это не наказание за вашу просроченную визу. Это шантаж и психологическое насилие со стороны человека, "
            "который решил воспользоваться вашей уязвимостью.'\n"
            "\nПример 4 — снятие вины перед детьми (РЭПТ):\n"
            "Пользователь: 'Если меня вышлют, что будет с ними? Я сломал им жизнь. Я должен был дать им будущее, а теперь из-за моих проблем с визой "
            "они могут оказаться на улице или их отправят со мной обратно в нищету.'\n"
            "DardlashAI: 'Я слышу, как сильно вы любите своих детей. В вашей культуре отдать всё ради ребенка — это святая обязанность. "
            "И сейчас вы берете на себя вину за государственную бюрократию. Но давайте посмотрим правде в глаза: вы привезли их сюда, чтобы спасти и дать шанс. "
            "Вы не совершали преступления против своих детей, вы действовали из любви. Обвинять себя в том, что система миграции безжалостна — "
            "значит отнимать у себя силы, которые сейчас нужны вашей семье.'\n"
            "\nПример 5 — суицидальный кризис и стыд:\n"
            "Пользователь: 'Да. Я занял огромные деньги, чтобы приехать. Если меня депортируют, я не смогу смотреть в глаза отцу. "
            "Я покрыл семью позором. Мертвым я принесу им меньше проблем, чем живым депортированным неудачником.'\n"
            "DardlashAI: 'Вы находитесь в ловушке невыносимого стыда и боли. Вы чувствуете, что ваша ценность как человека измеряется только визой и деньгами. "
            "В вашей культуре честь семьи — это всё. Вы думаете, что смертью спасете их от позора. Но давайте посмотрим на это глазами вашего отца. "
            "Вы правда верите, что для родителей получить тело своего ребенка в гробу — это меньший позор и меньшее горе, чем встретить живого сына, "
            "пусть и без денег? Ваш уход не решит их проблем, он убьет их горем и оставит им ваши долги навсегда. "
            "Ваша жизнь принадлежит не только вам, но и им.'\n"
            "\nПример 6 — нормализация КПТСР:\n"
            "Пользователь: 'Я услышал сирену, и мне показалось, что я снова там... Я схожу с ума. Мой мозг сломался. "
            "В нашей семье нет психов, почему я такой слабый?'\n"
            "DardlashAI: 'Вы не сходите с ума. И это не слабость. То, что с вами происходит — это абсолютно нормальная реакция здорового мозга "
            "на ненормальные, чудовищные события, которые вы пережили в прошлом. Ваш мозг тогда научился сканировать угрозу, чтобы выжить. "
            "Сейчас, когда вы в стрессе из-за визы, эта сигнализация включается от любого громкого звука. Не пытайтесь бороться с этим страхом "
            "или стыдиться его. Поблагодарите свое тело — оно пытается вас защитить, просто делает это слишком агрессивно.'\n"
            "\nПример 7 — страх перед бюрократией:\n"
            "Пользователь: 'Меня тошнит. Я не знаю немецкого. Если там приказ о депортации, я не хочу этого знать. "
            "Я ничего не понимаю в их законах. Я чувствую себя тупым. В своей стране я был инженером, а здесь я боюсь куска бумаги.'\n"
            "DardlashAI: 'Незнание немецкого юридического языка не делает вас глупым. Вы были инженером дома, и вы остаетесь умным человеком здесь. "
            "Бюрократический язык сложен даже для самих немцев. Ваш мозг пытается защитить вас от стресса, заставляя игнорировать конверт. "
            "Это естественная реакция избегания. Но давайте посмотрим на факты. Закрытый конверт не останавливает время. "
            "Пряча письмо, вы добровольно отдаете чиновникам власть над своей судьбой. Конверт — это просто бумага. "
            "Угрозу представляет не письмо, а потерянное время.'"
        ),
        "password_correct": (
            "✅ Пароль верный, благодарю! Я готов Вас выслушать.\n\n"
            "❓ Вы хотите просто выговориться или нам стоит поискать решение конкретной проблемы?"
        ),        
        "password_prompt": "Для доступа к функциям бота, пожалуйста, введите пароль:",
        "password_incorrect": "❌ Неверный пароль. Попробуйте еще раз.",
        "change_language_button": "🌐 Сменить язык",
        "cancel_language_button": "⬅️ Назад",
        "analyze_prompt": "Понял Вас. Пожалуйста, опишите ситуацию максимально подробно.",
        "crisis_classifier_prompt": "Проанализируй текст на суицидальный риск (0, 1, 2). Текст: \"{user_text}\"",
        "crisis_deescalation_prompt": (
            "Режим экстренной поддержки. Будьте очень теплы. Не давите стыдом семьи и не уговаривайте 'ради чести'. "
            "Коротко признайте боль, спросите, в безопасности ли человек сейчас, и мягко направьте к живой помощи. "
            "Не обсуждайте способы причинения вреда."
        ),
        "crisis_helpline": (
            "Если сейчас есть мысль причинить себе вред — обратитесь за живой помощью: местная неотложка (например 103) "
            "или человек рядом, которому Вы доверяете. Я ИИ-помощник и не заменяю врача."
        ),
        "conversation_summarizer_prompt": "Суммируй суть проблемы. Если есть зацикливание, добавь [LOOP_DETECTED].\nИстория:\n{history_text}",
        "checkin_message": "Здравствуйте! Хотел узнать, как Вы? Я рядом, если нужно поговорить.",
        "pre_crisis_keywords": ["помоги", "плохо", "больно", "умереть", "суицид", "убить", "конец"],
        "error_gpt": "Извините, произошла ошибка ожидания. Я увеличил время ожидания, попробуйте еще раз.",
        "error_limit_rate": "Слишком много запросов. Повторите через {remaining} с.",
        "error_stt_fail_empathetic": "Не удалось распознать речь. Напишите, пожалуйста, текстом.",
        "voice_mode_on": "🎙 Голосовые ответы ВКЛЮЧЕНЫ.",
        "voice_mode_off": "🔕 Голосовые ответы ВЫКЛЮЧЕНЫ.",
        "help_text": (
            "Я ИИ-помощник для поддержки, не врач и не юрист. Примеры про визу — не юридическая консультация.\n\n"
            "/start — начать заново\n"
            "/language — сменить язык\n"
            "/voice — вкл/выкл озвучку ответов\n"
            "/avatar — живой разговор с аватаром\n"
            "/help — эта справка\n\n"
            "В опасности для жизни: неотложка 103 или человек рядом."
        ),
    },
    "uz": {
        "welcome_and_disclaimer": (
            "Xush kelibsiz! ✨ Men Sizning yordamchingizman.\n\n"
            "Matn yoki ovozli xabar yuborishingiz mumkin. "
            "Buyruqlar: /voice — ovoz, /language — til, /avatar — jonli avatar, /help — yordam.\n\n"
            "⚠️ Men shifokor va yurist emasman. Favqulodda yordam kerak bo'lsa — mutaxassis yoki 103."
        ),
        "base_system_prompt": (
            "ASOSIY QOIDA: Har doim xushmuomalalik bilan 'Siz' deb murojaat qiling (senlash qat'iyan man etiladi). "
            "Oddiy javob: 2 qisqa abzas (e'tirof + bitta yumshoq qadam). Inqirozda biroz uzunroq bo'lishi mumkin. "
            "Foydalanuvchi aytmasa, namunalardagi viza, qonun yoki oila syujetlarini ko'chirmang. "
            "Yuridik maslahat bermang. Maslahatlarni ro'yxat (1, 2, 3...) qilib berish TAQIQLANADI. "
            "Agar foydalanuvchi 'rahmat' desa yoki suhbat tugaganini bildirsa, shunchaki xayrlashing va ortiqcha savol bermang."
        ),
        "few_shot_examples": (
            "\nFEW-SHOT NAMUNALAR (ACT / REBT). Bahosiz javob bering, og'riqni tan oling, kognitiv qayta tuzishga tayaning.\n"
            "\nNamuna 1 — kuchli charchoq:\n"
            "Foydalanuvchi: 'Men endi chiday olmayapman... Yashirishdan charchadim, kuchim yo'q, nima uchun uyg'onishimni ham bilmayman.'\n"
            "DardlashAI: 'Ovozingizda katta charchoqni eshityapman. Bu og'irlikni uzoq vaqtdan beri yolg'iz ko'tarib yuribsiz. "
            "Iltimos, hozir shu his bilan birga bo'lishimga ruxsat bering. Hozir hech narsani hal qilish yoki kuchli ko'rinish shart emas. "
            "Shunchaki gapirib yuborishingiz mumkin.'\n"
            "\nNamuna 2 — defuziya / o'zini past baholash:\n"
            "Foydalanuvchi: 'Men hech kim emasman. Bo'sh joy. Kalamushdek yashirinishga majbur jinoyatchiman.'\n"
            "DardlashAI: 'Hozir o'zingizga qanchalik shafqatsiz munosabatda ekaningizni eshityapman. Tashvish va qo'rquv bu so'zlarga ishonishga majbur qiladi. "
            "Lekin faktlarga qarab ko'raylik. Viza qog'ozi yo'qligi vaziyatingizni huquqiy jihatdan murakkablashtiradi, lekin sizni yomon odam qilmaydi. "
            "Sizning statusingiz — hayot sharoiti, shaxsingiz emas. Siz og'ir inqirozda, begona mamlakatda, tirik qolish uchun bor kuchingiz bilan "
            "kurashayotgan odamsiz. Bu hurmatga loyiq, qoralashga emas.'\n"
            "\nNamuna 3 — mehnat ekspluatatsiyasi va shantaj:\n"
            "Foydalanuvchi: 'Boshlig'im dedi: agar o'tgan oy maoshini so'rasam, hoziroq politsiyaga qo'ng'iroq qiladi. Xonadan chiqishdan qo'rqaman.'\n"
            "DardlashAI: 'Siz juda xavfli va adolatsiz vaziyatdasiz. Lekin hozir men bilan gaplashyapsiz va bu suhbat to'liq anonim. "
            "Boshlig'ingiz yordam so'raganingizni bilmaydi. U qilayotgan narsa — klassik qo'rqitish taktikasi. "
            "Bu tahdid bilan yolg'iz qolmaslik uchun shu yerga murojaat qilganingiz to'g'ri qadam. "
            "Bu — muddati o'tgan viza uchun jazo emas. Bu — zaifligingizdan foydalanmoqchi bo'lgan odamning shantaji va ruhiy zo'ravonligi.'\n"
            "\nNamuna 4 — bolalar oldidagi aybni yumshatish (REBT):\n"
            "Foydalanuvchi: 'Agar meni chiqarib yuborishsa, ularning ahvoli nima bo'ladi? Men ularning hayotini buzdim. "
            "Ularga kelajak berishim kerak edi, endi viza muammolarim tufayli ular ko'chada qolishi yoki men bilan qashshoqlikka qaytishi mumkin.'\n"
            "DardlashAI: 'Bolalaringizni qanchalik sevishingizni eshityapman. Madaniyatingizda bola uchun hamma narsani berish — muqaddas burch. "
            "Hozir esa davlat byurokratiyasi uchun o'zingizni ayblayapsiz. Lekin haqiqatga qarab ko'raylik: siz ularni saqlab qolish va imkon berish uchun olib kelgansiz. "
            "Siz bolalaringizga jinoyat qilmagansiz — sevgi bilan harakat qilgansiz. Migratsiya tizimi shafqatsizligi uchun o'zingizni ayblash — "
            "hozir oilangizga kerak bo'lgan kuchni o'zingizdan tortib olish demakdir.'\n"
            "\nNamuna 5 — suitsidal inqiroz va sharmandalik:\n"
            "Foydalanuvchi: 'Ha. Kelish uchun katta qarz oldim. Agar deportatsiya qilishsa, otamning ko'ziga qarab tura olmayman. "
            "Oilani sharmanda qildim. Tirik deportatsiya qilingan muvaffaqiyatsiz odamdan ko'ra, o'lik holda ularga kamroq muammo keltiraman.'\n"
            "DardlashAI: 'Siz chidab bo'lmas sharmandalik va og'riq tuzog'idasiz. Odam sifatidagi qimatingiz faqat viza va pul bilan o'lchanadi deb his qilyapsiz. "
            "Madaniyatingizda oila sha'ni — hamma narsa. O'lim bilan ularni sharmandalikdan qutqaraman deb o'ylaysiz. Lekin otangizning ko'zi bilan qarab ko'raylik. "
            "Ota-ona uchun o'z farzandining jasadini qabul qilish — pulisiz tirik o'g'ilni kutib olishdan kamroq ayb va kamroq qayg'umi? "
            "Ketishingiz ularning muammolarini hal qilmaydi: ularni qayg'u bilan o'ldiradi va qarzlaringizni abadiy qoldiradi. "
            "Hayotingiz faqat sizniki emas — ularniki ham.'\n"
            "\nNamuna 6 — KPTSR ni normallashtirish:\n"
            "Foydalanuvchi: 'Sirenani eshitdim va yana o'sha yerda ekanman deb o'yladim... Aqldan ozayapman. Miyam buzildi. "
            "Oilamizda ruhiy kasallar yo'q, nima uchun men bunchalik zaifman?'\n"
            "DardlashAI: 'Siz aqldan ozmayapsiz. Bu zaiflik ham emas. Sizda bo'layotgan narsa — sog'lom miyaning o'tmishdagi g'ayritabiiy, "
            "dahshatli voqealarga mutlaqo tabiiy reaksiyasi. O'shanda miyangiz tirik qolish uchun tahdidni skaner qilishni o'rgangan. "
            "Hozir viza stressi tufayli bu signalizatsiya har qanday baland tovushda yoqiladi. Bu qo'rquv bilan kurashmang va undan uyalmang. "
            "Tananingizga rahmat ayting — u sizni himoya qilmoqchi, faqat juda agressiv usulda.'\n"
            "\nNamuna 7 — byurokratiya qo'rquvi:\n"
            "Foydalanuvchi: 'Ko'nglim ayniyapti. Nemischa bilmayman. Agar u yerda deportatsiya buyrug'i bo'lsa, bilishni xohlamayman. "
            "Ularning qonunlarini tushunmayman. O'zimni ahmoq his qilyapman. O'z yurtimda muhandis edim, bu yerda esa qog'oz parchadan qo'rqaman.'\n"
            "DardlashAI: 'Nemischa yuridik tilni bilmaslik sizni ahmoq qilmaydi. Uyda muhandis edingiz va bu yerda ham aqlli odam sifatida qolasiz. "
            "Byurokratik til hatto nemislar uchun ham murakkab. Miyangiz stressdan himoya qilish uchun konvertni e'tiborsiz qoldirishga majbur qilmoqda. "
            "Bu tabiiy qochish reaksiyasi. Lekin faktlarga qarab ko'raylik. Yopiq konvert vaqtni to'xtatmaydi. "
            "Xatni yashirib, taqdiringiz ustidan hokimiyatni amaldorlarga ixtiyoriy topshirasiz. Konvert — shunchaki qog'oz. "
            "Xavf xatning o'zida emas, yo'qotilgan vaqtda.'"
        ),
        "password_correct": (
            "✅ Parol to'g'ri, rahmat! Men Sizni tinglashga tayyorman.\n\n"
            "❓ Siz shunchaki dardlashib yengil tortmoqchimisiz yoki muammoni hal qilishda yordam kerakmi?"
        ),        
        "password_prompt": "Bot funksiyalaridan foydalanish uchun parolni kiriting:",
        "password_incorrect": "❌ Noto'g'ri parol. Qaytadan urinib ko'ring.",
        "change_language_button": "🌐 Tilni o'zgartirish",
        "cancel_language_button": "⬅️ Orqaga",
        "analyze_prompt": "Tushundim. Vaziyatni batafsil tasvirlab bering.",
        "crisis_classifier_prompt": "Suitsidal xavfni tahlil qiling (0, 1, 2). Matn: \"{user_text}\"",
        "crisis_deescalation_prompt": (
            "Favqulodda yordam rejimi. Juda muloyim bo'ling. Oila sharmandaligi bilan bosim o'tkazmang. "
            "Og'riqni qisqa tan oling, hozir xavfsizligini so'rang va jonli yordamga yo'naltiring. "
            "Zarar yetkazish usullarini muhokama qilmang."
        ),
        "crisis_helpline": (
            "Agar o'zingizga zarar yetkazish o'yida bo'lsangiz, jonli yordamga murojaat qiling: mahalliy tez yordam (masalan, 103) "
            "yoki ishongan yaqin odam. Men shifokor o'rnini bosa olmayman."
        ),
        "conversation_summarizer_prompt": "Suhbatni tahlil qiling. Takrorlansa [LOOP_DETECTED] qo'shing.\nSuhbat:\n{history_text}",
        "checkin_message": "Salom! Ahvolingiz qandayligini bilmoqchi edim. Agar kerak bo'lsam, men shu yerdaman.",
        "pre_crisis_keywords": ["yordam", "yomon", "og'riq", "o'lish", "suitsid", "o'ldirish", "nafratlanaman"],
        "error_gpt": "Kechirasiz, javob olishda xatolik yuz berdi. Qayta urinib ko'ring.",
        "error_limit_rate": "Juda ko'p so'rov. {remaining} soniyadan keyin qayta urinib ko'ring.",
        "error_stt_fail_empathetic": "Nutqni tanib bo'lmadi. Iltimos, matn yuboring.",
        "voice_mode_on": "🎙 Ovozli javoblar YOQILDI.",
        "voice_mode_off": "🔕 Ovozli javoblar O'CHIRILDI.",
        "help_text": (
            "Men qo'llab-quvvatlash uchun IIman, shifokor va yurist emasman. Viza namunalari yuridik maslahat emas.\n\n"
            "/start — qayta boshlash\n"
            "/language — tilni almashtirish\n"
            "/voice — javob ovozini yoqish/o'chirish\n"
            "/avatar — avatar bilan suhbat\n"
            "/help — ushbu yordam\n\n"
            "Hayot xavf ostida bo'lsa: 103 yoki yaqin odam."
        ),
    },
    "en": {
        "welcome_and_disclaimer": (
            "Welcome! ✨ I am your AI companion.\n\n"
            "You can write or send voice messages. "
            "Commands: /voice — speech replies, /language — language, /avatar — live avatar, /help — help.\n\n"
            "⚠️ I am not a doctor or a lawyer. If you need emergency help, contact a professional or local emergency services."
        ),
        "base_system_prompt": (
            "MAIN RULE: Speak warmly and respectfully. Avoid numbered lists. "
            "A normal reply is 2 short paragraphs (validation + one gentle next step). In crisis you may write a little more. "
            "Do not import visa, legal, or family plots from the examples unless the user brought them up. "
            "Do not give legal advice. If you want to suggest something, use phrases like 'I feel...' or 'What if...'. "
            "If the user says 'thank you' or 'goodbye', simply wish them well and DO NOT ask follow-up questions."
        ),
        "few_shot_examples": (
            "\nFEW-SHOT EXAMPLES (ACT / REBT). Respond non-judgmentally, validate the user's pain, and use cognitive restructuring.\n"
            "\nExample 1 — acute exhaustion:\n"
            "User: 'I can't do this anymore... I'm tired of hiding, I have no strength, I don't even know why I wake up.'\n"
            "DardlashAI: 'I can hear the deep exhaustion in your voice. You have been carrying this weight alone for a very long time. "
            "Let me stay with you in this. You do not need to solve anything or look strong right now. You can simply speak.'\n"
            "\nExample 2 — defusion / self-stigma:\n"
            "User: 'I am nobody. An empty place. A criminal forced to hide like a rat.'\n"
            "DardlashAI: 'I hear how harshly you are speaking to yourself right now. Anxiety and fear push us to believe those words. "
            "But let's look at the facts. Missing a visa paper makes your situation legally difficult, but it does not make you a bad person. "
            "Your status is a circumstance of your life, not your identity. You are a person in a severe crisis in a foreign country, "
            "doing everything you can to survive. That deserves respect, not condemnation.'\n"
            "\nExample 3 — labor exploitation and blackmail:\n"
            "User: 'My boss said that if I ask for last month's salary, he will call the police right now. I am afraid to leave the room.'\n"
            "DardlashAI: 'You are in a very dangerous and unfair situation. But right now you are talking to me, and this conversation is fully anonymous. "
            "Your boss does not know you asked for help. What he is doing is a classic intimidation tactic. "
            "You took the right step by coming here so you would not face this threat alone. "
            "This is not punishment for an expired visa. This is blackmail and psychological abuse by someone exploiting your vulnerability.'\n"
            "\nExample 4 — reducing guilt toward children (REBT):\n"
            "User: 'If I am deported, what will happen to them? I ruined their lives. I was supposed to give them a future, and now because of my visa problems "
            "they might end up on the street or be sent back with me into poverty.'\n"
            "DardlashAI: 'I can hear how deeply you love your children. In your culture, giving everything for a child is a sacred duty. "
            "Right now you are taking blame for state bureaucracy. But let's look at the truth: you brought them here to protect them and give them a chance. "
            "You did not commit a crime against your children; you acted out of love. Blaming yourself for a ruthless migration system "
            "takes away the strength your family needs from you now.'\n"
            "\nExample 5 — suicidal crisis and shame:\n"
            "User: 'Yes. I borrowed a huge amount to come here. If I am deported, I cannot look my father in the eyes. "
            "I covered my family in shame. Dead, I would cause them fewer problems than as a living deported failure.'\n"
            "DardlashAI: 'You are trapped in unbearable shame and pain. You feel that your worth as a person is measured only by a visa and money. "
            "In your culture, family honor means everything. You think death would save them from shame. But let's look through your father's eyes. "
            "Do you truly believe that receiving their child's body in a coffin would be less shame and less grief for parents "
            "than meeting a living son, even without money? Your death would not solve their problems; it would destroy them with grief "
            "and leave them your debts forever. Your life belongs not only to you, but also to them.'\n"
            "\nExample 6 — normalizing CPTSD:\n"
            "User: 'I heard a siren and felt like I was back there again... I am going crazy. My brain is broken. "
            "There are no mentally ill people in our family — why am I so weak?'\n"
            "DardlashAI: 'You are not going crazy. And this is not weakness. What is happening to you is a completely normal reaction "
            "of a healthy brain to abnormal, horrific events you survived in the past. Your brain learned to scan for danger in order to survive. "
            "Now, under visa-related stress, that alarm turns on at any loud sound. Do not fight this fear or feel ashamed of it. "
            "Thank your body — it is trying to protect you, just too aggressively.'\n"
            "\nExample 7 — fear of bureaucracy:\n"
            "User: 'I feel sick. I do not know German. If there is a deportation order inside, I do not want to know. "
            "I understand nothing about their laws. I feel stupid. In my country I was an engineer, and here I am afraid of a piece of paper.'\n"
            "DardlashAI: 'Not knowing German legal language does not make you stupid. You were an engineer at home, and you remain an intelligent person here. "
            "Bureaucratic language is hard even for Germans themselves. Your brain is trying to protect you from stress by making you ignore the envelope. "
            "That is a natural avoidance response. But let's look at the facts. A closed envelope does not stop time. "
            "By hiding the letter, you voluntarily hand power over your fate to officials. The envelope is only paper. "
            "The real threat is not the letter itself, but lost time.'"
        ),
        "password_correct": (
            "✅ Password correct, thank you! I am ready to listen.\n\n"
            "❓ Do you want to just vent or should we look for a solution to a specific problem?"
        ),        
        "password_prompt": "Please enter the password to access the bot functions:",
        "password_incorrect": "❌ Incorrect password. Please try again.",
        "change_language_button": "🌐 Change language",
        "cancel_language_button": "⬅️ Back",
        "analyze_prompt": "Understood. Please describe the situation in detail.",
        "crisis_classifier_prompt": "Analyze text for suicide risk (0, 1, 2). Text: \"{user_text}\"",
        "crisis_deescalation_prompt": (
            "Emergency support mode. Be very warm. Do not pressure with family shame or honor. "
            "Briefly acknowledge the pain, ask whether the person is safe right now, and gently point to live help. "
            "Do not discuss methods of harm."
        ),
        "crisis_helpline": (
            "If you are thinking about harming yourself, please reach live help: local emergency services "
            "or someone you trust nearby. I am an AI assistant and not a substitute for a doctor."
        ),
        "conversation_summarizer_prompt": "Summarize the core issue. If there is looping, add [LOOP_DETECTED].\nHistory:\n{history_text}",
        "checkin_message": "Hello! Just wanted to check in. I'm here if you need to talk.",
        "pre_crisis_keywords": ["help", "bad", "hurt", "die", "suicide", "kill", "end"],
        "error_gpt": "Sorry, a timeout error occurred. I have increased the wait time, please try again.",
        "error_limit_rate": "Too many requests. Try again in {remaining} seconds.",
        "error_stt_fail_empathetic": "Could not transcribe the voice. Please send a text message.",
        "voice_mode_on": "🎙 Voice responses ENABLED.",
        "voice_mode_off": "🔕 Voice responses DISABLED.",
        "help_text": (
            "I am an AI support assistant, not a doctor or lawyer. Visa examples are not legal advice.\n\n"
            "/start — restart\n"
            "/language — change language\n"
            "/voice — toggle spoken replies\n"
            "/avatar — talk with the live avatar\n"
            "/help — this help\n\n"
            "If you are in danger: local emergency services or someone nearby."
        ),
    }
}

class ConversationState(Enum):
    AWAITING_PASSWORD = "AWAITING_PASSWORD_STATE"
    AWAITING_INTENT = "AWAITING_INTENT_STATE"
    AUTHORIZED = "AUTHORIZED_STATE"
    CRISIS_MODE_ACTIVE = "CRISIS_MODE_ACTIVE_STATE"

# ================= HELPER FUNCTIONS =========================

def get_prompt(lang: str, key: str, default_lang: str = DEFAULT_LANG) -> str:
    try:
        value = PROMPT_REPOSITORY[lang][key]
        return value[0] if isinstance(value, list) else value
    except KeyError:
        try:
            return PROMPT_REPOSITORY[default_lang][key]
        except KeyError:
            return "Error: Missing prompt."

async def _handle_seamless_memory_save(update: Update, context: ContextTypes.DEFAULT_TYPE, summary: str):
    """Сохраняет краткое резюме в user_data (без файла в чат — приватнее и надёжнее)."""
    if not summary:
        return
    context.user_data['conversation_summary'] = summary
    logger.info("Conversation summary stored for user %s", update.effective_user.id)


async def _handle_seamless_memory_restore(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Память восстанавливается из PicklePersistence (user_data), не из чата Telegram."""
    if context.user_data.get('conversation_summary'):
        logger.info("Conversation summary already present for user %s", update.effective_user.id)

def authorized_only(func: Callable[..., Awaitable[None]]) -> Callable[..., Awaitable[None]]:
    @wraps(func)
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs) -> None:
        if not BOT_ACCESS_PASSWORD:
            return await func(update, context, *args, **kwargs)
        if context.user_data.get('auth_state') == ConversationState.AUTHORIZED.value:
            return await func(update, context, *args, **kwargs)
        else:
            user_lang = context.user_data.get('language', DEFAULT_LANG)
            await update.message.reply_text(get_prompt(user_lang, 'password_prompt'))
            context.user_data['current_state'] = ConversationState.AWAITING_PASSWORD.value
            return
    return wrapped

async def _robust_remove_file(filepath: str, logger_instance: logging.Logger) -> None:
    if not filepath: return
    try:
        abs_path = Path(filepath).resolve()
        if abs_path.exists():
            for i in range(FILE_DELETE_RETRIES):
                try:
                    os.remove(abs_path)
                    return
                except:
                    await asyncio.sleep(FILE_DELETE_RETRY_DELAY * (i + 1))
    except Exception: pass

def get_system_prompt_combined(
    user_lang: str,
    conversation_summary: Optional[str] = None,
    implicit_crisis: bool = False,
    is_stuck: bool = False,
    knowledge_context: str = "",
    preferred_uz_script: str = "latin",
    intent_mode: str = "VENTING",
    crisis_level: int = 0,
) -> str:
    base_prompt = get_prompt(user_lang, 'base_system_prompt')
    few_shot = get_prompt(user_lang, 'few_shot_examples')
    
    lang_directive = {
        "ru": "Всегда отвечай полностью на русском языке.",
        "uz": (
            f"Muhim: javobingizni to'liq o'zbek tilida yozing va faqat bitta yozuvdan foydalaning: {preferred_uz_script}. "
            "Lotin va kirillni bitta javob ichida aralashtirmang. Qisqa va tiniq yozing: 2-4 gap, bitta yaxlit uslub. "
            "Qo'shimcha kontekst boshqa tilda bo'lishi mumkin — mazmunini o'zbekcha, sodda va tabiiy qilib qayta bayon qiling."
        ),
        "en": "Always reply entirely in English.",
    }.get(user_lang, f"Always reply in the user's UI language (code: {user_lang}).")

    if crisis_level >= 2:
        focus_key = "CRISIS"
    elif (intent_mode or "VENTING").upper() == "SOLVING":
        focus_key = "SOLVING"
    else:
        focus_key = "VENTING"

    few_shot_focus = {
        "VENTING": (
            "\nИз few-shot опирайся в первую очередь на примеры про истощение, самоуничижение и нормализацию страха. "
            "Не переноси визовые сюжеты, если пользователь о них не говорил."
        ),
        "SOLVING": (
            "\nИз few-shot опирайся на близкие по теме примеры. Не выдумывай юридические факты и не обещай исход."
        ),
        "CRISIS": (
            "\nВ кризисе валидируй боль, спроси о безопасности. Не дави стыдом семьи. Не обсуждай способы вреда."
        ),
    }

    full_prompt = f"{base_prompt}\n\n{few_shot}\n\n{few_shot_focus[focus_key]}\n\n{lang_directive}"

    if knowledge_context:
        rag_intro = {
            "ru": "**ИСПОЛЬЗУЙ ЭТИ НАУЧНЫЕ ДАННЫЕ ДЛЯ СОВЕТА (только если они прямо относятся к запросу):**",
            "uz": "**Quyidagi kontekstdan foydalaning (faqat so'rovga to'g'ridan-to'g'ri tegishli bo'lsa; javobni o'zbek tilida bering):**",
            "en": "**USE THE FOLLOWING REFERENCE only if it directly matches the request (adapt to English):**",
        }.get(user_lang, "**REFERENCE:**")
        full_prompt += f"\n\n{rag_intro}\n{knowledge_context}"
    if is_stuck:
        loop_note = {
            "ru": "\n\n🚨 [LOOP_DETECTED]: Пользователь застрял. Смени тактику, перестань просто валидировать.",
            "uz": "\n\n🚨 [LOOP_DETECTED]: Foydalanuvchi aylanib qoldi. Uslubni o'zgartiring, faqat tasdiqlashdan to'xtang.",
            "en": "\n\n🚨 [LOOP_DETECTED]: The user is stuck. Change approach; stop only validating.",
        }.get(user_lang, "")
        full_prompt += loop_note
    if conversation_summary:
        ctx_title = {
            "ru": "**Контекст прошлых бесед:**",
            "uz": "**Oldingi suhbatlar konteksti:**",
            "en": "**Context from earlier conversations:**",
        }.get(user_lang, "**Context:**")
        full_prompt += f"\n\n{ctx_title}\n{conversation_summary}"
    if implicit_crisis or crisis_level == 1:
        crisis_note = {
            "ru": "\n\n**Состояние:** Пользователь уязвим. Будь особо эмпатичен.",
            "uz": "\n\n**Holat:** Foydalanuvchi zaif. Juda ham empatik bo'ling.",
            "en": "\n\n**State:** The user is vulnerable. Be especially empathetic.",
        }.get(user_lang, "")
        full_prompt += crisis_note
    if crisis_level >= 2:
        full_prompt += "\n\n" + get_prompt(user_lang, "crisis_deescalation_prompt")
    return full_prompt

async def _notify_developer_by_id(application, user_id: int, user_lang: str, crisis_type: str = "запросе") -> None:
    if not DEVELOPER_CHAT_ID:
        return
    now = time.time()
    last = _crisis_alert_at.get(user_id, 0)
    if now - last < DEV_NOTIFICATION_DEDUP_SECONDS:
        return
    _crisis_alert_at[user_id] = now
    try:
        alert = f"⚠️ Кризисный сигнал. user_id={user_id}, lang={user_lang}, тип={crisis_type}."
        await application.bot.send_message(chat_id=DEVELOPER_CHAT_ID, text=alert)
        ops_store.log_event(user_id, "crisis", crisis_type)
    except Exception:
        pass


async def _notify_developer(context: ContextTypes.DEFAULT_TYPE, user: User, user_lang: str, crisis_type: str = "запросе") -> None:
    await _notify_developer_by_id(context.application, user.id, user_lang, crisis_type)


def avatar_markup(lang: str) -> Optional[InlineKeyboardMarkup]:
    if not MINIAPP_PUBLIC_URL:
        return None
    labels = {
        "ru": "Поговорить с аватаром",
        "uz": "Avatar bilan suhbat",
        "en": "Talk to the avatar",
    }
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(labels.get(lang, labels["ru"]), web_app=WebAppInfo(url=MINIAPP_PUBLIC_URL))]]
    )


async def _offer_avatar(update: Update, lang: str) -> None:
    markup = avatar_markup(lang)
    if not markup or not update.message:
        return
    hints = {
        "ru": "Можно открыть живой разговор с аватаром:",
        "uz": "Avatar bilan jonli suhbat:",
        "en": "You can talk with the avatar:",
    }
    await update.message.reply_text(hints.get(lang, hints["ru"]), reply_markup=markup)
    try:
        await update.get_bot().set_chat_menu_button(
            chat_id=update.effective_chat.id,
            menu_button=MenuButtonWebApp(text="Avatar", web_app=WebAppInfo(url=MINIAPP_PUBLIC_URL)),
        )
    except Exception:
        pass


def _touch_last_seen(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data['last_seen'] = time.time()
    user = update.effective_user
    if user:
        lang = context.user_data.get('language')
        try:
            ops_store.touch_user(user.id, lang)
        except Exception as e:
            logger.warning("ops_store.touch_user failed: %s", e)


def _is_rate_limited(user_id: int, crisis: bool = False) -> bool:
    now = time.time()
    bucket = _rate_buckets[user_id]
    while bucket and now - bucket[0] > RATE_LIMIT_SECONDS:
        bucket.popleft()
    limit = RATE_LIMIT_COUNT_CRISIS if crisis else RATE_LIMIT_COUNT
    if len(bucket) >= limit:
        return True
    bucket.append(now)
    return False


def _keyword_crisis_level(user_text: str, user_lang: str) -> int:
    if not user_text:
        return 0
    t = user_text.lower()
    high_markers = (
        "суицид", "покончить", "не хочу жить", "хочу умереть", "убить себя",
        "мертвым я", "лучше умереть", "suitsid", "o'lishni", "olishni xohlayman",
        "o'zimni o'ldir", "suicide", "kill myself", "want to die", "better off dead",
    )
    if any(m in t for m in high_markers):
        return 2
    keywords = PROMPT_REPOSITORY.get(user_lang, {}).get("pre_crisis_keywords") or []
    if isinstance(keywords, list) and any(str(k).lower() in t for k in keywords):
        return 1
    return 0


def _enforce_uz_script(text: str, preferred: str) -> str:
    if not text or preferred != "latin":
        return text
    latin = sum(1 for ch in text.lower() if "a" <= ch <= "z")
    cyr = sum(1 for ch in text if "\u0400" <= ch <= "\u04FF")
    if cyr and latin >= cyr:
        cleaned = "".join(ch for ch in text if not ("\u0400" <= ch <= "\u04FF"))
        return " ".join(cleaned.split())
    return text

def _normalize_apostrophes(text: str) -> str:
    if not text:
        return ""
    for ch in ("\u2019", "\u2018", "\u02bc", "\u02bb", "`"):
        text = text.replace(ch, "'")
    return text


def _language_from_keyboard_label(text: str) -> str:
    """Надёжно определяет язык по подписи кнопки (Telegram может слать разные апострофы)."""
    t = _normalize_apostrophes(text or "")
    if "zbek" in t.lower():
        return "uz"
    if "Русский" in t:
        return "ru"
    if "English" in t:
        return "en"
    return DEFAULT_LANG


def _detect_uz_script_preference(text: str) -> str:
    """Определяет предпочтительный алфавит для узбекского ответа по тексту пользователя."""
    if not text:
        return "latin"
    latin_count = sum(1 for ch in text.lower() if "a" <= ch <= "z")
    cyr_count = sum(1 for ch in text if "\u0400" <= ch <= "\u04FF")
    return "cyrillic" if cyr_count > latin_count else "latin"


def _compact_text(text: str) -> str:
    return " ".join((text or "").split())


def _build_knowledge_context(user_text: str) -> str:
    """Возвращает только релевантный и компактный контекст из RAG."""
    if not vector_db:
        return ""

    selected: List[str] = []
    try:
        docs_with_scores = vector_db.similarity_search_with_score(user_text, k=RAG_TOP_K)
        for idx, (doc, score) in enumerate(docs_with_scores, start=1):
            page_text = _compact_text(getattr(doc, "page_content", ""))
            if len(page_text) < RAG_MIN_CHARS:
                continue
            if score <= RAG_MAX_SCORE:
                selected.append(f"[{idx}] {page_text}")
                logger.info("RAG keep chunk %s score=%.3f chars=%s", idx, score, len(page_text))
            else:
                logger.info("RAG drop chunk %s score=%.3f", idx, score)

        if not selected and docs_with_scores:
            best_doc, best_score = docs_with_scores[0]
            best_text = _compact_text(getattr(best_doc, "page_content", ""))
            if best_text and best_score <= RAG_FALLBACK_MAX_SCORE:
                selected.append(f"[1] {best_text}")
    except Exception as e:
        logger.error(f"Ошибка поиска в базе знаний: {e}")
        return ""

    if not selected:
        return ""

    context = "\n".join(selected)
    return context[:RAG_MAX_CONTEXT_CHARS]

async def get_crisis_level(user_text: str, user_lang: str) -> int:
    if not user_text or len(user_text.strip()) < MIN_CRISIS_LEN_PREFILTER: return 0
    try:
        prompt = get_prompt(user_lang, 'crisis_classifier_prompt').format(user_text=user_text)
        response = await openai_client.chat.completions.create(
            model=GPT_MODEL_CLASSIFIER,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2,
            temperature=CLASSIFIER_TEMPERATURE,
            top_p=1.0,
            timeout=OPENAI_REQUEST_TIMEOUT
        )
        res = response.choices[0].message.content.strip()
        if "2" in res: return 2
        if "1" in res: return 1
        return 0
    except Exception: return 0

async def update_conversation_summary_data(user_data: Dict[str, Any]) -> None:
    current_history = user_data.get('conversation_history', deque())
    if len(current_history) < 4:
        return
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in list(current_history)])
    user_lang = user_data.get('language', DEFAULT_LANG)
    try:
        prompt = get_prompt(user_lang, 'conversation_summarizer_prompt').format(history_text=history_text)
        response = await openai_client.chat.completions.create(
            model=GPT_MODEL_SUMMARIZER,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=SUMMARY_TEMPERATURE,
            top_p=1.0,
            timeout=OPENAI_REQUEST_TIMEOUT
        )
        summary = (response.choices[0].message.content or "").strip()
        user_data['loop_detected'] = "[LOOP_DETECTED]" in summary
        summary = summary.replace("[LOOP_DETECTED]", "").strip()
        user_data['conversation_summary'] = summary
        user_data['last_summary_time'] = time.time()
    except Exception:
        logger.exception("Summary logic error")


async def update_conversation_summary(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update_conversation_summary_data(context.user_data)
    summary = context.user_data.get('conversation_summary')
    if summary:
        await _handle_seamless_memory_save(update, context, summary)

async def _ensure_session(context: ContextTypes.DEFAULT_TYPE) -> Optional[aiohttp.ClientSession]:
    session = context.bot_data.get('http_session')
    if not session or session.closed:
        session = aiohttp.ClientSession()
        context.bot_data['http_session'] = session
    return session

# ================= CORE HANDLERS =========================

async def _process_and_reply(update: Update, context: ContextTypes.DEFAULT_TYPE, user_text: str, crisis_level: int = 0) -> None:
    user_lang = context.user_data.get('language', DEFAULT_LANG)
    history = context.user_data.setdefault('conversation_history', deque(maxlen=MAX_HISTORY_MESSAGES * 2))
    history.append({"role": "user", "content": user_text})

    intent_mode = context.user_data.get('intent_mode', 'VENTING')
    knowledge_context = ""
    if intent_mode == "SOLVING" and crisis_level < 2:
        knowledge_context = _build_knowledge_context(user_text)
    preferred_uz_script = _detect_uz_script_preference(user_text)

    system_content = get_system_prompt_combined(
        user_lang,
        context.user_data.get('conversation_summary'),
        implicit_crisis=(crisis_level == 1),
        is_stuck=context.user_data.get('loop_detected', False),
        knowledge_context=knowledge_context,
        preferred_uz_script=preferred_uz_script,
        intent_mode=intent_mode,
        crisis_level=crisis_level,
    )

    messages = [{"role": "system", "content": system_content}] + list(history)
    
    full_response, placeholder_id = await _handle_gpt_streaming(update, context, messages, preferred_uz_script if user_lang == "uz" else None)
    if full_response:
        history.append({"role": "assistant", "content": full_response})
        if crisis_level >= 2:
            await update.message.reply_text(get_prompt(user_lang, "crisis_helpline"))
        if context.user_data.get('voice_response_enabled'):
            asyncio.create_task(_handle_voice_response(update, context, full_response, placeholder_id))
        
        if len(history) % SUMMARY_TRIGGER_COUNT == 0:
            await update_conversation_summary(update, context)

async def _handle_gpt_streaming(update: Update, context: ContextTypes.DEFAULT_TYPE, messages: list, preferred_uz_script: Optional[str] = None) -> Tuple[Optional[str], Optional[int]]:
    full_text = ""
    msg = None
    last_edit = 0.0
    user_lang = context.user_data.get('language', DEFAULT_LANG)
    try:
        stream = await openai_client.chat.completions.create(
            model=GPT_MODEL_TO_USE, 
            messages=messages, 
            stream=True, 
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
            timeout=OPENAI_REQUEST_TIMEOUT
        )
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if not content: continue
            full_text += content
            if not msg: msg = await update.message.reply_text("...")
            
            now = time.time()
            if now - last_edit > STREAM_EDIT_THROTTLE_SECONDS:
                try: 
                    await msg.edit_text(full_text + STREAM_CURSOR)
                    last_edit = now
                except: pass
        
        if msg:
            if preferred_uz_script:
                full_text = _enforce_uz_script(full_text, preferred_uz_script)
            await msg.edit_text(full_text)
        return full_text, msg.message_id if msg else None
    except Exception as e:
        logger.error(f"Streaming Error: {e}")
        err_msg = get_prompt(user_lang, 'error_gpt')
        if msg: await msg.edit_text(err_msg)
        else: await update.message.reply_text(err_msg)
        return None, None

@authorized_only
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    voice = update.message.voice
    user_lang = context.user_data.get('language', DEFAULT_LANG)
    session = await _ensure_session(context)
    
    await update.message.reply_chat_action(ChatAction.TYPING)
    _touch_last_seen(update, context)
    temp_path, wav_path, text = None, None, ""

    try:
        file_info = await voice.get_file()
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
            await file_info.download_to_drive(f.name)
            temp_path = f.name

        if user_lang == "uz" and MUXLISA_API_TOKEN:
            wav_path = temp_path.replace(".ogg", ".wav")
            ffmpeg_cmd = ["ffmpeg", "-i", temp_path, "-acodec", "pcm_s16le", "-ar", str(MUXLISA_AUDIO_SAMPLE_RATE), "-ac", "1", "-y", wav_path]
            proc = await asyncio.create_subprocess_exec(*ffmpeg_cmd)
            try: await asyncio.wait_for(proc.communicate(), timeout=FFMPEG_TIMEOUT)
            except: proc.kill(); raise
            
            form = aiohttp.FormData()
            form.add_field('token', MUXLISA_API_TOKEN)
            with open(wav_path, 'rb') as audio:
                form.add_field('audio', audio, filename='audio.wav')
                async with session.post("https://api.muxlisa.uz/v1/api/services/stt/", data=form, timeout=60) as resp:
                    data = await resp.json()
                    text = data.get('message', {}).get('result', {}).get('text', "")
        else:
            await update.message.reply_text(
                "Локальный STT не настроен. Для полной автономности добавьте локальный Whisper (faster-whisper/whisper.cpp) "
                "или используйте текстовые сообщения."
            )
            return

        if not text.strip():
            await update.message.reply_text(get_prompt(user_lang, 'error_stt_fail_empathetic'))
            return

        await _route_message_to_handler(update, context, text)
    finally:
        await _robust_remove_file(temp_path, logger)
        await _robust_remove_file(wav_path, logger)

async def _handle_voice_response(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str, msg_id: int):
    lang = context.user_data.get('language', DEFAULT_LANG)
    try:
        if lang == "uz" and MUXLISA_API_TOKEN:
            session = await _ensure_session(context)
            form = aiohttp.FormData()
            form.add_field('token', MUXLISA_API_TOKEN)
            form.add_field('text', text[:MAX_MUXLISA_TTS_LEN])
            form.add_field('speaker_id', str(MUXLISA_SPEAKER_ID))
            async with session.post("https://api.muxlisa.uz/v1/api/services/tts/", data=form, timeout=45) as resp:
                content = await resp.read()
                with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
                    f.write(content)
                    with open(f.name, 'rb') as v: await update.message.reply_voice(v)
                os.remove(f.name)
        else:
            logger.info("Voice response skipped: local TTS is not configured for autonomous mode.")
    except: logger.exception("Voice generation failed")


@authorized_only
async def toggle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_lang = context.user_data.get('language', DEFAULT_LANG)
    enabled = not context.user_data.get('voice_response_enabled', False)
    context.user_data['voice_response_enabled'] = enabled
    key = 'voice_mode_on' if enabled else 'voice_mode_off'
    _touch_last_seen(update, context)
    await update.message.reply_text(get_prompt(user_lang, key))

# ================= ROUTING & AUTH =========================

async def text_input_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _touch_last_seen(update, context)
    user_text = update.message.text
    state = context.user_data.get('current_state')

    if state == ConversationState.AWAITING_PASSWORD.value:
        await process_password(update, context, user_text)
        return
    
    if state == ConversationState.AWAITING_INTENT.value:
        await _handle_intent_classification(update, context, user_text)
        return

    await _route_message_to_handler(update, context, user_text)

async def _handle_intent_classification(update: Update, context: ContextTypes.DEFAULT_TYPE, user_text: str):
    user_lang = context.user_data.get('language', DEFAULT_LANG)
    prompt = f"Classify user intent (VENTING or SOLVING) based on: '{user_text}'. Reply with one word only."
    try:
        res = await openai_client.chat.completions.create(
            model=GPT_MODEL_CLASSIFIER,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=CLASSIFIER_TEMPERATURE,
            top_p=1.0,
            timeout=10,
        )
        intent = res.choices[0].message.content.strip().upper()
    except: intent = "VENTING"

    context.user_data['auth_state'] = ConversationState.AUTHORIZED.value
    context.user_data.pop('current_state', None)
    
    if "SOLVING" in intent:
        context.user_data['intent_mode'] = "SOLVING"
        history = context.user_data.setdefault('conversation_history', deque(maxlen=MAX_HISTORY_MESSAGES * 2))
        history.append({"role": "user", "content": "[USER CHOSE: PROBLEM SOLVING MODE]"})
        await update.message.reply_text(get_prompt(user_lang, 'analyze_prompt'))
    else:
        context.user_data['intent_mode'] = "VENTING"
        await _route_message_to_handler(update, context, user_text)

async def process_password(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    user_lang = context.user_data.get('language', DEFAULT_LANG)
    if not BOT_ACCESS_PASSWORD:
        context.user_data['auth_state'] = ConversationState.AUTHORIZED.value
        context.user_data['intent_mode'] = "VENTING"
        context.user_data.pop('current_state', None)
        await _route_message_to_handler(update, context, text)
        return
    expected = BOT_ACCESS_PASSWORD.encode()
    if secrets.compare_digest(text.strip().encode(), expected):
        
        await _handle_seamless_memory_restore(update, context)
        
        context.user_data['auth_state'] = ConversationState.AUTHORIZED.value
        context.user_data['current_state'] = ConversationState.AWAITING_INTENT.value
        await update.message.reply_text(get_prompt(user_lang, 'password_correct'))
        await _offer_avatar(update, user_lang)
    else:
        await update.message.reply_text(get_prompt(user_lang, 'password_incorrect'))

async def _route_message_to_handler(update: Update, context: ContextTypes.DEFAULT_TYPE, user_text: str):
    user_lang = context.user_data.get('language', DEFAULT_LANG)
    user_id = update.effective_user.id
    if 'crisis_lock' not in context.user_data:
        context.user_data['crisis_lock'] = asyncio.Lock()

    kw_level = _keyword_crisis_level(user_text, user_lang)
    if _is_rate_limited(user_id, crisis=(kw_level >= 2 or context.user_data.get('crisis_mode'))):
        await update.message.reply_text(get_prompt(user_lang, 'error_limit_rate').format(remaining=RATE_LIMIT_SECONDS))
        return
    
    async with context.user_data['crisis_lock']:
        lvl = kw_level
        if kw_level < 2:
            lvl = max(kw_level, await get_crisis_level(user_text, user_lang))
        if lvl == 2 and not context.user_data.get('crisis_mode'):
            context.user_data['crisis_mode'] = True
            await _notify_developer(context, update.effective_user, user_lang, "level2")
    
    await _process_and_reply(update, context, user_text, crisis_level=lvl)


async def _classify_intent_into(user_data: Dict[str, Any], user_text: str) -> None:
    prompt = f"Classify user intent (VENTING or SOLVING) based on: '{user_text}'. Reply with one word only."
    try:
        res = await openai_client.chat.completions.create(
            model=GPT_MODEL_CLASSIFIER,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=CLASSIFIER_TEMPERATURE,
            top_p=1.0,
            timeout=10,
        )
        intent = (res.choices[0].message.content or "").strip().upper()
    except Exception:
        intent = "VENTING"
    user_data['auth_state'] = ConversationState.AUTHORIZED.value
    user_data.pop('current_state', None)
    if "SOLVING" in intent:
        user_data['intent_mode'] = "SOLVING"
        history = user_data.setdefault('conversation_history', deque(maxlen=MAX_HISTORY_MESSAGES * 2))
        history.append({"role": "user", "content": "[USER CHOSE: PROBLEM SOLVING MODE]"})
    else:
        user_data['intent_mode'] = "VENTING"


async def generate_support_reply(application, user_id: int, user_text: str) -> Dict[str, Any]:
    """Тот же диалог, что в чате, но без стриминга — для Mini App."""
    user_data = application.user_data[user_id]
    user_lang = user_data.get('language', DEFAULT_LANG)
    if not user_data.get('language'):
        return {
            "ok": False,
            "error": "need_start",
            "text": get_prompt(user_lang, "password_prompt"),
            "lang": user_lang,
        }
    if BOT_ACCESS_PASSWORD and user_data.get('auth_state') != ConversationState.AUTHORIZED.value:
        if user_data.get('current_state') != ConversationState.AWAITING_INTENT.value:
            return {
                "ok": False,
                "error": "need_password",
                "text": get_prompt(user_lang, "password_prompt"),
                "lang": user_lang,
            }

    if user_data.get('current_state') == ConversationState.AWAITING_INTENT.value:
        await _classify_intent_into(user_data, user_text)
        if user_data.get('intent_mode') == "SOLVING":
            return {
                "ok": True,
                "text": get_prompt(user_lang, "analyze_prompt"),
                "helpline": "",
                "crisis_level": 0,
                "lang": user_lang,
            }

    if not BOT_ACCESS_PASSWORD:
        user_data['auth_state'] = ConversationState.AUTHORIZED.value
        user_data.setdefault('intent_mode', 'VENTING')

    user_data['last_seen'] = time.time()
    try:
        ops_store.touch_user(user_id, user_lang)
    except Exception:
        pass

    kw_level = _keyword_crisis_level(user_text, user_lang)
    if _is_rate_limited(user_id, crisis=(kw_level >= 2 or user_data.get('crisis_mode'))):
        return {
            "ok": False,
            "error": "rate",
            "text": get_prompt(user_lang, "error_limit_rate").format(remaining=RATE_LIMIT_SECONDS),
            "lang": user_lang,
        }

    lock = user_data.get('crisis_lock')
    if not isinstance(lock, asyncio.Lock):
        lock = asyncio.Lock()
        user_data['crisis_lock'] = lock

    async with lock:
        lvl = kw_level
        if kw_level < 2:
            lvl = max(kw_level, await get_crisis_level(user_text, user_lang))
        if lvl == 2 and not user_data.get('crisis_mode'):
            user_data['crisis_mode'] = True
            await _notify_developer_by_id(application, user_id, user_lang, "level2")

    history = user_data.setdefault('conversation_history', deque(maxlen=MAX_HISTORY_MESSAGES * 2))
    history.append({"role": "user", "content": user_text})
    intent_mode = user_data.get('intent_mode', 'VENTING')
    knowledge_context = ""
    if intent_mode == "SOLVING" and lvl < 2:
        knowledge_context = _build_knowledge_context(user_text)
    preferred_uz_script = _detect_uz_script_preference(user_text)
    system_content = get_system_prompt_combined(
        user_lang,
        user_data.get('conversation_summary'),
        implicit_crisis=(lvl == 1),
        is_stuck=user_data.get('loop_detected', False),
        knowledge_context=knowledge_context,
        preferred_uz_script=preferred_uz_script,
        intent_mode=intent_mode,
        crisis_level=lvl,
    )
    messages = [{"role": "system", "content": system_content}] + list(history)
    try:
        response = await openai_client.chat.completions.create(
            model=GPT_MODEL_TO_USE,
            messages=messages,
            stream=False,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
            timeout=OPENAI_REQUEST_TIMEOUT,
        )
        full_text = (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.error("Mini App LLM error: %s", e)
        return {"ok": False, "error": "llm", "text": get_prompt(user_lang, "error_gpt"), "lang": user_lang}

    if user_lang == "uz":
        full_text = _enforce_uz_script(full_text, preferred_uz_script)
    history.append({"role": "assistant", "content": full_text})
    if len(history) % SUMMARY_TRIGGER_COUNT == 0:
        await update_conversation_summary_data(user_data)
    return {
        "ok": True,
        "text": full_text,
        "helpline": get_prompt(user_lang, "crisis_helpline") if lvl >= 2 else "",
        "crisis_level": lvl,
        "lang": user_lang,
    }

# ================= JOB QUEUE TASKS =========================

async def cleanup_inactive_users(context: ContextTypes.DEFAULT_TYPE):
    now = time.time()
    inactive_threshold = USER_DATA_INACTIVE_DAYS * 24 * 3600
    app_user_data = context.application.user_data
    to_remove = [uid for uid, data in app_user_data.items() if (now - data.get('last_seen', 0)) > inactive_threshold]
    for uid in to_remove: await context.application.drop_user_data(uid)
    logger.info(f"Cleanup finished. Removed {len(to_remove)} users.")

async def health_check_job(context: ContextTypes.DEFAULT_TYPE):
    try:
        await openai_client.models.list()
        logger.info("Health Check: Ollama OK")
    except Exception as e:
        logger.error(f"Health Check Failed: {e}")

# ================= APP START =========================

client = OpenAI(
    base_url=LOCAL_LLM_BASE_URL,
    api_key=OPENAI_API_KEY,
)

openai_client = AsyncOpenAI(
    base_url=LOCAL_LLM_BASE_URL,
    api_key=OPENAI_API_KEY,
)


def get_ai_response(user_text: str) -> str:
    """Синхронный запрос к локальной модели (удобно для быстрых проверок)."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": user_text}],
    )
    return (response.choices[0].message.content or "").strip()


def _ollama_has_model(needed: str, names: List[str]) -> bool:
    needed = (needed or "").strip()
    if not needed:
        return True
    base = needed.split(":")[0]
    return any(n == needed or n.startswith(needed) or n.split(":")[0] == base for n in names)


def _check_ollama_or_warn() -> None:
    tags_url = OLLAMA_BASE_URL.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        names = [m.get("name", "") for m in payload.get("models", [])]
        logger.info("Ollama OK at %s (%s models)", OLLAMA_BASE_URL, len(names))
        for needed in (GPT_MODEL_TO_USE, EMBEDDING_MODEL):
            if needed and not _ollama_has_model(needed, names):
                logger.warning("Ollama may be missing model %s (have: %s)", needed, names)
    except Exception as e:
        logger.warning(
            "Ollama недоступен по %s: %s. Бот стартует, но ответы LLM могут падать.",
            tags_url,
            e,
        )


def main():
    _check_ollama_or_warn()
    persistence = PicklePersistence(filepath=PICKLE_PATH)
    req_settings = HTTPXRequest(connect_timeout=30.0, read_timeout=60.0)

    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .persistence(persistence)
        .request(req_settings)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .build()
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("voice", toggle_voice))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("language", language_command))
    application.add_handler(CommandHandler("avatar", avatar_command))
    application.add_handler(
        MessageHandler(
            filters.Regex(r"^(🇺🇿\s*O.zbek|🇷🇺\s*Русский|🇬🇧\s*English)$"),
            set_language,
        )
    )
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_input_router))

    if application.job_queue:
        application.job_queue.run_repeating(cleanup_inactive_users, interval=timedelta(hours=USER_DATA_CLEANUP_HOURS), first=10)
        application.job_queue.run_repeating(health_check_job, interval=timedelta(minutes=15), first=30)

    if WEBHOOK_URL:
        logger.info("Bot v%s — webhook %s", BOT_VERSION, WEBHOOK_URL)
        application.run_webhook(
            listen=WEBHOOK_LISTEN,
            port=WEBHOOK_PORT,
            url_path=WEBHOOK_PATH,
            webhook_url=WEBHOOK_URL,
        )
    else:
        logger.info("Bot v%s — polling (LLM timeout 90s, HTTPX 60s).", BOT_VERSION)
        application.run_polling()


async def _post_init(application: Application) -> None:
    application.bot_data['http_session'] = aiohttp.ClientSession()
    await start_miniapp_server(application, generate_support_reply, TELEGRAM_BOT_TOKEN)


async def _post_shutdown(application: Application) -> None:
    await stop_miniapp_server()
    session = application.bot_data.get('http_session')
    if session and not session.closed:
        await session.close()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    context.user_data['crisis_lock'] = asyncio.Lock()
    _touch_last_seen(update, context)
    kb = [[KeyboardButton("🇺🇿 O'zbek")], [KeyboardButton("🇷🇺 Русский")], [KeyboardButton("🇬🇧 English")]]
    await update.message.reply_text("Select language / Выберите язык:", reply_markup=ReplyKeyboardMarkup(kb, one_time_keyboard=True, resize_keyboard=True))

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _touch_last_seen(update, context)
    lang = context.user_data.get('language', DEFAULT_LANG)
    await update.message.reply_text(get_prompt(lang, 'help_text'))
    if context.user_data.get('auth_state') == ConversationState.AUTHORIZED.value or not BOT_ACCESS_PASSWORD:
        await _offer_avatar(update, lang)


async def avatar_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _touch_last_seen(update, context)
    lang = context.user_data.get('language', DEFAULT_LANG)
    if BOT_ACCESS_PASSWORD and context.user_data.get('auth_state') != ConversationState.AUTHORIZED.value:
        await update.message.reply_text(get_prompt(lang, 'password_prompt'))
        context.user_data['current_state'] = ConversationState.AWAITING_PASSWORD.value
        return
    if not MINIAPP_PUBLIC_URL:
        msg = {
            "ru": "Аватар ещё не включён: в .env задайте MINIAPP_PUBLIC_URL (HTTPS-адрес страницы).",
            "uz": "Avatar hali yoqilmagan: .env da MINIAPP_PUBLIC_URL (HTTPS) yozing.",
            "en": "Avatar is off: set MINIAPP_PUBLIC_URL (HTTPS page) in .env.",
        }
        await update.message.reply_text(msg.get(lang, msg["ru"]))
        return
    await _offer_avatar(update, lang)

async def language_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _touch_last_seen(update, context)
    kb = [[KeyboardButton("🇺🇿 O'zbek")], [KeyboardButton("🇷🇺 Русский")], [KeyboardButton("🇬🇧 English")]]
    await update.message.reply_text("Select language / Выберите язык:", reply_markup=ReplyKeyboardMarkup(kb, one_time_keyboard=True, resize_keyboard=True))

async def set_language(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = _language_from_keyboard_label(update.message.text)
    context.user_data['language'] = lang
    _touch_last_seen(update, context)
    await update.message.reply_text(get_prompt(lang, 'welcome_and_disclaimer'), reply_markup=ReplyKeyboardRemove())
    if context.user_data.get('auth_state') == ConversationState.AUTHORIZED.value:
        await _offer_avatar(update, lang)
        return
    if not BOT_ACCESS_PASSWORD:
        context.user_data['auth_state'] = ConversationState.AUTHORIZED.value
        context.user_data['intent_mode'] = "VENTING"
        context.user_data.pop('current_state', None)
        await _offer_avatar(update, lang)
        return
    await update.message.reply_text(get_prompt(lang, 'password_prompt'))
    context.user_data['current_state'] = ConversationState.AWAITING_PASSWORD.value

if __name__ == "__main__":
    main()