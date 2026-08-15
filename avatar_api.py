# -*- coding: utf-8 -*-
"""Mini App: статика аватара + API хода диалога (тот же мозг, что у бота)."""
import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import parse_qsl

import aiohttp
from aiohttp import web

logger = logging.getLogger(__name__)

MINIAPP_DIR = Path(__file__).resolve().parent / "miniapp"
MINIAPP_ENABLED = os.getenv("MINIAPP_ENABLED", "1").strip() not in ("0", "false", "False")
MINIAPP_LISTEN = os.getenv("MINIAPP_LISTEN", "0.0.0.0")
MINIAPP_PORT = int(os.getenv("MINIAPP_PORT", "8090"))
MINIAPP_PUBLIC_URL = (os.getenv("MINIAPP_PUBLIC_URL") or "").strip()
AVATAR_GLB_URL = os.getenv(
    "AVATAR_GLB_URL",
    "https://models.readyplayer.me/64bfa15f0e72c63d7c3934a6.glb"
    "?morphTargets=ARKit,Oculus+Visemes,mouthOpen,mouthSmile,eyesClosed,eyesLookUp,eyesLookDown"
    "&textureSizeLimit=512&textureFormat=png",
)
INITDATA_MAX_AGE = int(os.getenv("MINIAPP_INITDATA_MAX_AGE", "86400"))

_runner: Optional[web.AppRunner] = None
_generate_reply: Optional[Callable[..., Any]] = None
_bot_token = ""
_muxlisa_token = ""
_muxlisa_speaker = "1"
_http: Optional[aiohttp.ClientSession] = None


def validate_init_data(init_data: str, bot_token: str) -> Dict[str, Any]:
    if not init_data or not bot_token:
        raise ValueError("empty initData")
    parsed = dict(parse_qsl(init_data, keep_blank_values=True))
    received_hash = parsed.pop("hash", "")
    if not received_hash:
        raise ValueError("no hash")
    data_check = "\n".join(f"{k}={v}" for k, v in sorted(parsed.items()))
    secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    calculated = hmac.new(secret, data_check.encode(), hashlib.sha256).hexdigest()
    if not secrets.compare_digest(calculated, received_hash):
        raise ValueError("bad hash")
    auth_date = int(parsed.get("auth_date") or "0")
    if auth_date and time.time() - auth_date > INITDATA_MAX_AGE:
        raise ValueError("expired")
    user = json.loads(parsed.get("user") or "{}")
    if not user.get("id"):
        raise ValueError("no user")
    return user


def _init_data_from_request(request: web.Request) -> str:
    return (
        request.headers.get("X-Telegram-Init-Data")
        or request.rel_url.query.get("initData")
        or ""
    )


async def _user_from_request(request: web.Request) -> Dict[str, Any]:
    return validate_init_data(_init_data_from_request(request), _bot_token)


async def _ffmpeg_to_wav(src: str, dst: str) -> None:
    cmd = [
        "ffmpeg", "-i", src,
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", dst,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
    )
    try:
        await asyncio.wait_for(proc.communicate(), timeout=45)
    except asyncio.TimeoutError:
        proc.kill()
        raise
    if proc.returncode not in (0, None) and not Path(dst).exists():
        raise RuntimeError("ffmpeg failed")


async def muxlisa_stt(wav_path: str) -> str:
    if not _muxlisa_token or _http is None:
        return ""
    form = aiohttp.FormData()
    form.add_field("token", _muxlisa_token)
    with open(wav_path, "rb") as audio:
        form.add_field("audio", audio, filename="audio.wav")
        async with _http.post(
            "https://api.muxlisa.uz/v1/api/services/stt/", data=form, timeout=60
        ) as resp:
            data = await resp.json(content_type=None)
    return (
        data.get("message", {}).get("result", {}).get("text", "")
        if isinstance(data, dict)
        else ""
    )


async def muxlisa_tts_wav(text: str) -> Optional[bytes]:
    if not _muxlisa_token or _http is None or not text:
        return None
    form = aiohttp.FormData()
    form.add_field("token", _muxlisa_token)
    form.add_field("text", text[:510])
    form.add_field("speaker_id", _muxlisa_speaker)
    async with _http.post(
        "https://api.muxlisa.uz/v1/api/services/tts/", data=form, timeout=45
    ) as resp:
        raw = await resp.read()
    if not raw or len(raw) < 64:
        return None
    src = dst = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
            f.write(raw)
            src = f.name
        dst = src + ".wav"
        await _ffmpeg_to_wav(src, dst)
        return Path(dst).read_bytes()
    except Exception as e:
        logger.warning("TTS wav convert failed: %s", e)
        return raw
    finally:
        for p in (src, dst):
            if p:
                try:
                    os.remove(p)
                except OSError:
                    pass


async def handle_index(_request: web.Request) -> web.FileResponse:
    return web.FileResponse(MINIAPP_DIR / "index.html")


async def handle_session(request: web.Request) -> web.Response:
    try:
        tg_user = await _user_from_request(request)
    except ValueError as e:
        return web.json_response({"ok": False, "error": "auth", "detail": str(e)}, status=401)

    application = request.app["tg_app"]
    user_id = int(tg_user["id"])
    ud = application.user_data[user_id]
    lang = ud.get("language") or "ru"
    authorized = (not os.getenv("BOT_ACCESS_PASSWORD")) or ud.get("auth_state") == "AUTHORIZED_STATE"
    return web.json_response(
        {
            "ok": True,
            "authorized": bool(authorized),
            "lang": lang,
            "user_id": user_id,
            "tts": "muxlisa" if (_muxlisa_token and lang == "uz") else "browser",
            "avatar_url": AVATAR_GLB_URL,
        }
    )


async def handle_turn(request: web.Request) -> web.Response:
    try:
        tg_user = await _user_from_request(request)
    except ValueError as e:
        return web.json_response({"ok": False, "error": "auth", "detail": str(e)}, status=401)
    if _generate_reply is None:
        return web.json_response({"ok": False, "error": "not_ready"}, status=503)

    try:
        payload = await request.json()
    except Exception:
        payload = {}
    text = (payload.get("text") or "").strip()
    if not text:
        return web.json_response({"ok": False, "error": "empty"}, status=400)

    user_id = int(tg_user["id"])
    result = await _generate_reply(request.app["tg_app"], user_id, text)
    if result.get("ok") and result.get("lang") == "uz":
        wav = await muxlisa_tts_wav(result.get("text") or "")
        if wav:
            result["audio_b64"] = base64.b64encode(wav).decode("ascii")
            result["audio_mime"] = "audio/wav"
            result["tts"] = "muxlisa"
        else:
            result["tts"] = "browser"
    elif result.get("ok"):
        result["tts"] = "browser"
    return web.json_response(result, status=200 if result.get("ok") else 400)


async def handle_stt(request: web.Request) -> web.Response:
    try:
        tg_user = await _user_from_request(request)
    except ValueError as e:
        return web.json_response({"ok": False, "error": "auth", "detail": str(e)}, status=401)

    application = request.app["tg_app"]
    user_id = int(tg_user["id"])
    lang = application.user_data[user_id].get("language") or "ru"
    if lang != "uz" or not _muxlisa_token:
        return web.json_response(
            {"ok": False, "error": "stt_unavailable", "use_browser": True},
            status=400,
        )

    reader = await request.multipart()
    field = await reader.next()
    if field is None:
        return web.json_response({"ok": False, "error": "no_file"}, status=400)
    src = dst = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                f.write(chunk)
            src = f.name
        dst = src + ".wav"
        await _ffmpeg_to_wav(src, dst)
        text = (await muxlisa_stt(dst) or "").strip()
        if not text:
            return web.json_response({"ok": False, "error": "empty_stt"}, status=400)
        return web.json_response({"ok": True, "text": text})
    except Exception as e:
        logger.exception("Mini App STT failed")
        return web.json_response({"ok": False, "error": str(e)}, status=500)
    finally:
        for p in (src, dst):
            if p:
                try:
                    os.remove(p)
                except OSError:
                    pass


def _build_app(application) -> web.Application:
    app = web.Application(client_max_size=12 * 1024 * 1024)
    app["tg_app"] = application
    app.router.add_get("/", handle_index)
    app.router.add_get("/index.html", handle_index)
    app.router.add_get("/api/session", handle_session)
    app.router.add_post("/api/turn", handle_turn)
    app.router.add_post("/api/stt", handle_stt)
    app.router.add_get("/styles.css", lambda r: web.FileResponse(MINIAPP_DIR / "styles.css"))
    app.router.add_get("/app.js", lambda r: web.FileResponse(MINIAPP_DIR / "app.js"))
    return app


async def start_miniapp_server(application, generate_reply, bot_token: str) -> None:
    global _runner, _generate_reply, _bot_token, _muxlisa_token, _muxlisa_speaker, _http
    if not MINIAPP_ENABLED:
        logger.info("Mini App disabled (MINIAPP_ENABLED=0)")
        return
    if not MINIAPP_DIR.joinpath("index.html").exists():
        logger.warning("Mini App: нет %s — сервер не стартует", MINIAPP_DIR)
        return
    _generate_reply = generate_reply
    _bot_token = bot_token
    _muxlisa_token = os.getenv("MUXLISA_API_TOKEN") or ""
    _muxlisa_speaker = os.getenv("MUXLISA_SPEAKER_ID", "1")
    _http = application.bot_data.get("http_session")
    if _http is None or _http.closed:
        _http = aiohttp.ClientSession()
        application.bot_data["http_session"] = _http

    try:
        runner = web.AppRunner(_build_app(application), access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, MINIAPP_LISTEN, MINIAPP_PORT)
        await site.start()
        _runner = runner
        logger.info(
            "Mini App listening on %s:%s (public %s)",
            MINIAPP_LISTEN,
            MINIAPP_PORT,
            MINIAPP_PUBLIC_URL or "не задан — кнопки в Telegram не будет",
        )
    except OSError as e:
        logger.error("Mini App port %s busy or failed: %s", MINIAPP_PORT, e)


async def stop_miniapp_server() -> None:
    global _runner
    if _runner:
        await _runner.cleanup()
        _runner = None
