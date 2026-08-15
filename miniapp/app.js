import { TalkingHead } from "talkinghead";

const ui = {
  ru: {
    loading: "Загрузка аватара…",
    ready: "Можно говорить или написать.",
    wait: "Думаю…",
    needStart: "Сначала в чате бота: /start → язык → пароль.",
    authFail: "Не удалось войти. Откройте Mini App из Telegram.",
    sttFail: "Не разобрал речь. Напишите текстом.",
    rec: "Слушаю… отпустите кнопку",
    placeholder: "Напишите сюда…",
    send: "Отправить",
  },
  uz: {
    loading: "Avatar yuklanmoqda…",
    ready: "Gapiring yoki yozing.",
    wait: "O‘ylayapman…",
    needStart: "Avval botda: /start → til → parol.",
    authFail: "Kirib bo‘lmadi. Mini Appni Telegramdan oching.",
    sttFail: "Nutqni tushunmadim. Matn yozing.",
    rec: "Tinglayapman… tugmani qo‘yib yuboring",
    placeholder: "Shu yerga yozing…",
    send: "Yuborish",
  },
  en: {
    loading: "Loading avatar…",
    ready: "Speak or type.",
    wait: "Thinking…",
    needStart: "First in the bot chat: /start → language → password.",
    authFail: "Could not sign in. Open the Mini App from Telegram.",
    sttFail: "Could not hear that. Please type.",
    rec: "Listening… release the button",
    placeholder: "Type here…",
    send: "Send",
  },
};

const tg = window.Telegram && window.Telegram.WebApp;
if (tg) {
  tg.ready();
  tg.expand();
  tg.setHeaderColor("#1a1f2e");
  tg.setBackgroundColor("#1a1f2e");
}

const statusEl = document.getElementById("status");
const replyEl = document.getElementById("reply");
const helplineEl = document.getElementById("helpline");
const textEl = document.getElementById("text");
const sendBtn = document.getElementById("send");
const micBtn = document.getElementById("mic");

let head = null;
let session = null;
let busy = false;
let recorder = null;
let chunks = [];

function t(key) {
  const lang = (session && session.lang) || "ru";
  return (ui[lang] || ui.ru)[key];
}

function initData() {
  return (tg && tg.initData) || "";
}

async function api(path, options = {}) {
  const headers = Object.assign({ "X-Telegram-Init-Data": initData() }, options.headers || {});
  const res = await fetch(path, Object.assign({}, options, { headers }));
  const data = await res.json().catch(() => ({ ok: false }));
  return data;
}

function timingsFromText(text, durationSec) {
  const words = (text || "").trim().split(/\s+/).filter(Boolean);
  const total = Math.max(400, durationSec * 1000);
  if (!words.length) {
    return { words: [text || ""], wtimes: [0], wdurations: [total] };
  }
  const slot = total / words.length;
  return {
    words,
    wtimes: words.map((_, i) => i * slot),
    wdurations: words.map(() => slot * 0.92),
  };
}

async function speakWithAudio(text, b64, mime) {
  if (!head || !b64) return false;
  const raw = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const buf = await ctx.decodeAudioData(raw.buffer.slice(0));
  const timing = timingsFromText(text, buf.duration);
  head.speakAudio({
    audio: buf,
    words: timing.words,
    wtimes: timing.wtimes,
    wdurations: timing.wdurations,
  });
  return true;
}

function speakBrowser(text, lang) {
  if (!window.speechSynthesis) return;
  const u = new SpeechSynthesisUtterance(text);
  u.lang = lang === "uz" ? "uz-UZ" : lang === "en" ? "en-US" : "ru-RU";
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(u);
  if (!head) return;
  const words = text.trim().split(/\s+/);
  const slot = 280;
  try {
    head.speakAudio({
      words,
      wtimes: words.map((_, i) => i * slot),
      wdurations: words.map(() => slot * 0.85),
    });
  } catch (_e) {
    /* avatar may still idle; voice still plays */
  }
}

async function loadAvatar(url) {
  const node = document.getElementById("avatar");
  head = new TalkingHead(node, {
    ttsEndpoint: "https://invalid.local/tts",
    lipsyncModules: ["en"],
    cameraView: "head",
    cameraRotateEnable: false,
    cameraPanEnable: false,
  });
  await head.showAvatar({
    url,
    body: "F",
    avatarMood: "neutral",
    lipsyncLang: "en",
  });
}

async function sendText(text) {
  const value = (text || textEl.value || "").trim();
  if (!value || busy) return;
  busy = true;
  sendBtn.disabled = true;
  statusEl.textContent = t("wait");
  try {
    const data = await api("api/turn", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: value }),
    });
    if (!data.ok) {
      statusEl.textContent = data.text || t("needStart");
      return;
    }
    textEl.value = "";
    replyEl.textContent = data.text || "";
    if (data.helpline) {
      helplineEl.hidden = false;
      helplineEl.textContent = data.helpline;
    } else {
      helplineEl.hidden = true;
    }
    statusEl.textContent = t("ready");
    let spoken = false;
    if (data.audio_b64) {
      try {
        spoken = await speakWithAudio(data.text, data.audio_b64, data.audio_mime);
      } catch (e) {
        console.warn(e);
      }
    }
    if (!spoken) speakBrowser(data.text, data.lang || session.lang);
  } catch (e) {
    statusEl.textContent = String(e);
  } finally {
    busy = false;
    sendBtn.disabled = false;
  }
}

function browserSttAvailable() {
  return window.SpeechRecognition || window.webkitSpeechRecognition;
}

function listenBrowser() {
  const Ctor = browserSttAvailable();
  if (!Ctor) return false;
  const rec = new Ctor();
  rec.lang = session.lang === "uz" ? "uz-UZ" : session.lang === "en" ? "en-US" : "ru-RU";
  rec.interimResults = false;
  rec.onresult = (ev) => {
    const said = ev.results[0][0].transcript;
    textEl.value = said;
    sendText(said);
  };
  rec.onerror = () => {
    statusEl.textContent = t("sttFail");
  };
  rec.start();
  statusEl.textContent = t("rec");
  return true;
}

async function startRecord() {
  if (busy) return;
  if (session.lang !== "uz") {
    if (!listenBrowser()) statusEl.textContent = t("sttFail");
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    chunks = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (e) => {
      if (e.data.size) chunks.push(e.data);
    };
    recorder.start();
    micBtn.classList.add("rec");
    statusEl.textContent = t("rec");
  } catch (_e) {
    statusEl.textContent = t("sttFail");
  }
}

async function stopRecord() {
  micBtn.classList.remove("rec");
  if (!recorder) return;
  const rec = recorder;
  recorder = null;
  await new Promise((resolve) => {
    rec.onstop = resolve;
    rec.stop();
    rec.stream.getTracks().forEach((tr) => tr.stop());
  });
  const blob = new Blob(chunks, { type: rec.mimeType || "audio/webm" });
  const form = new FormData();
  form.append("audio", blob, "speech.webm");
  statusEl.textContent = t("wait");
  const data = await api("api/stt", { method: "POST", body: form });
  if (!data.ok || !data.text) {
    statusEl.textContent = t("sttFail");
    return;
  }
  textEl.value = data.text;
  sendText(data.text);
}

async function boot() {
  statusEl.textContent = ui.ru.loading;
  if (!initData()) {
    statusEl.textContent = ui.ru.authFail;
    return;
  }
  session = await api("api/session");
  textEl.placeholder = t("placeholder");
  if (!session.ok) {
    statusEl.textContent = t("authFail");
    return;
  }
  if (!session.authorized) {
    statusEl.textContent = t("needStart");
    return;
  }
  statusEl.textContent = t("loading");
  try {
    await loadAvatar(session.avatar_url);
    statusEl.textContent = t("ready");
  } catch (e) {
    console.warn(e);
    statusEl.textContent = t("ready");
  }
}

sendBtn.addEventListener("click", () => sendText());
textEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendText();
  }
});
micBtn.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  startRecord();
});
micBtn.addEventListener("pointerup", () => stopRecord());
micBtn.addEventListener("pointercancel", () => stopRecord());
document.addEventListener("visibilitychange", () => {
  if (!head) return;
  if (document.visibilityState === "visible") head.start();
  else head.stop();
});

boot();
