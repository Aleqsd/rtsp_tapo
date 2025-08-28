import sys
import time
import os
import glob
import base64
import csv
import cv2
import requests
import argparse
from PIL import Image
import numpy as np
from datetime import datetime, timedelta

# Try to use IANA timezone "Europe/Paris" (fallback to system local time if unavailable)
try:
    from zoneinfo import ZoneInfo

    TZ = ZoneInfo(os.getenv("TZ", "Europe/Paris"))
except Exception:
    TZ = None  # fallback to naive localtime

from openai import OpenAI

# ================== ENV CONFIG ==================
# Required
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY") or os.getenv("OPENAI_API_KEY")
RTSP_TAPO_USER = os.getenv("RTSP_TAPO_USER")
RTSP_TAPO_PASSWORD = os.getenv("RTSP_TAPO_PASSWORD")
RTSP_TAPO_IP = os.getenv("RTSP_TAPO_IP")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Optional
RTSP_TAPO_PORT = os.getenv("RTSP_TAPO_PORT", "554")
RTSP_TAPO_STREAM = os.getenv("RTSP_TAPO_STREAM", "stream1")  # stream1 or stream2
ANALYZE_AFTER = float(os.getenv("ANALYZE_AFTER", "5"))  # seconds before capture
INTERVAL_SECONDS = int(os.getenv("INTERVAL_SECONDS", "3600"))  # default 1h
NO_FFMPEG = os.getenv("NO_FFMPEG", "0").lower() in ("1", "true", "yes")

# Quiet hours window (inclusive start, exclusive end)
QUIET_START_HOUR = int(os.getenv("QUIET_START_HOUR", "22"))  # 22:00
QUIET_END_HOUR = int(os.getenv("QUIET_END_HOUR", "8"))  # 08:00

# History & weekly report
HISTORY_PATH = os.getenv("HISTORY_PATH", "history.csv")
WEEKLY_REPORT_DAY = int(
    os.getenv("WEEKLY_REPORT_DAY", "6")
)  # 0=Mon .. 6=Sun (default Sunday)
WEEKLY_REPORT_HOUR = int(os.getenv("WEEKLY_REPORT_HOUR", "20"))  # 24h format
WEEKLY_REPORT_MINUTE = int(os.getenv("WEEKLY_REPORT_MINUTE", "0"))
LAST_REPORT_MARK = os.getenv(
    "LAST_REPORT_MARK", ".last_weekly_report"
)  # file to prevent duplicates

# Check required env vars
missing = [
    k
    for k, v in {
        "OPEN_AI_API_KEY": OPEN_AI_API_KEY,
        "RTSP_TAPO_USER": RTSP_TAPO_USER,
        "RTSP_TAPO_PASSWORD": RTSP_TAPO_PASSWORD,
        "RTSP_TAPO_IP": RTSP_TAPO_IP,
        "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
        "TELEGRAM_CHAT_ID": TELEGRAM_CHAT_ID,
    }.items()
    if not v
]
if missing:
    print(
        f"[ERROR] Missing environment variables: {', '.join(missing)}", file=sys.stderr
    )
    sys.exit(1)

# OpenAI client
_openai_client = OpenAI(api_key=OPEN_AI_API_KEY)


# ================== HELPERS ==================
def now_tz() -> datetime:
    """Get current datetime in configured timezone (or local naive if TZ missing)."""
    return datetime.now(TZ) if TZ else datetime.now()


def in_quiet_hours(dt: datetime) -> bool:
    """Return True if current time is within [QUIET_START_HOUR, next day QUIET_END_HOUR)."""
    h = dt.hour
    if QUIET_START_HOUR <= h or h < QUIET_END_HOUR:
        return True
    return False


def seconds_until_quiet_end(dt: datetime) -> int:
    """Compute seconds until the next QUIET_END_HOUR (today if not passed, else tomorrow)."""
    target = dt.replace(hour=QUIET_END_HOUR, minute=0, second=0, microsecond=0)
    if dt.hour >= QUIET_END_HOUR:
        target = target + timedelta(days=1)
    return max(1, int((target - dt).total_seconds()))


def send_telegram_status(level: str, image_path: str | None = None):
    """
    Send a Telegram notification indicating the detected level.
    level ∈ {"empty","low","ok","full"}.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(
            "[WARN] Telegram env vars not set; skipping notification.", file=sys.stderr
        )
        return

    # Friendly caption per level
    emoji = {"empty": "🚨", "low": "⚠️", "ok": "✅", "full": "🟢"}.get(level, "ℹ️")
    human = {
        "empty": "Bol vide — il faut remettre des croquettes",
        "low": "Niveau bas — à surveiller / ajouter bientôt",
        "ok": "Assez de croquettes",
        "full": "Bol bien rempli",
    }.get(level, f"État: {level}")
    caption = f"{emoji} {human} 🐾"

    try:
        if image_path and os.path.isfile(image_path):
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            with open(image_path, "rb") as f:
                files = {"photo": f}
                data = {
                    "chat_id": TELEGRAM_CHAT_ID,
                    "caption": caption,
                    "disable_notification": False,
                }
                r = requests.post(url, data=data, files=files, timeout=20)
        else:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            r = requests.post(
                url,
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": caption,
                    "disable_notification": False,
                },
                timeout=20,
            )

        r.raise_for_status()
        print("[INFO] Telegram notification sent.", file=sys.stderr)
    except Exception as e:
        print(f"[WARN] Telegram notification failed: {e}", file=sys.stderr)


def send_telegram_error(err_msg: str):
    """Send a Telegram notification for an error (used with anti-spam latch)."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    msg = f"⚠️ Croquettes counter error:\n{err_msg[:800]}"
    try:
        r = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": msg,
                "disable_notification": False,
            },
            timeout=20,
        )
        r.raise_for_status()
        print("[INFO] Telegram ERROR notification sent.", file=sys.stderr)
    except Exception as e:
        print(f"[WARN] Telegram error notification failed: {e}", file=sys.stderr)


def send_telegram_message(text: str):
    """Utility to send a plain text Telegram message."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        r = requests.post(
            url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=20
        )
        r.raise_for_status()
        print("[INFO] Telegram message sent.", file=sys.stderr)
    except Exception as e:
        print(f"[WARN] Telegram message failed: {e}", file=sys.stderr)


def build_rtsp_url(ip: str, port: str, user: str, password: str, stream: str) -> str:
    """Build RTSP URL with credentials."""
    auth = f"{user}:{password}@" if user and password else ""
    return f"rtsp://{auth}{ip}:{port}/{stream}"


def open_capture(rtsp_url: str, use_ffmpeg: bool = True) -> cv2.VideoCapture:
    """Open RTSP stream with OpenCV, using FFmpeg backend if available."""
    backend = cv2.CAP_FFMPEG if use_ffmpeg else cv2.CAP_ANY
    cap = cv2.VideoCapture(rtsp_url, backend)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 25)
    return cap


def bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR frame to PIL RGB Image."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_data_url(pil_img: Image.Image) -> str:
    """Convert PIL image to base64 data URL."""
    import io

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _save_and_prune_last_images(
    pil_img: Image.Image, out_dir: str = "last_images", keep: int = 3
) -> str:
    """
    Save the current PIL image into `out_dir` with a timestamped filename,
    then prune to keep only the newest `keep` images (delete older ones).
    Returns the saved path.
    """
    _ensure_dir(out_dir)
    ts = int(time.time())
    filename = f"frame_{ts}.png"
    path = os.path.join(out_dir, filename)

    try:
        pil_img.save(path)
        print(f"[INFO] Saved debug image: {path}", file=sys.stderr)
    except Exception as e:
        print(f"[WARN] Failed to save debug image: {e}", file=sys.stderr)
        return ""

    try:
        files = sorted(
            glob.glob(os.path.join(out_dir, "*.png")), key=lambda p: os.path.getmtime(p)
        )
        if len(files) > keep:
            to_remove = files[0 : len(files) - keep]
            for old in to_remove:
                try:
                    os.remove(old)
                    print(f"[INFO] Pruned old image: {old}", file=sys.stderr)
                except Exception as e:
                    print(
                        f"[WARN] Failed to remove old image {old}: {e}", file=sys.stderr
                    )
    except Exception as e:
        print(f"[WARN] Prune step failed: {e}", file=sys.stderr)

    return path


# ================== VISION: RICHER LEVEL DETECTION ==================
def kibbles_level_with_gpt5_nano(pil_img: Image.Image) -> tuple[str, float]:
    """
    Return (level, confidence) where:
      level ∈ {"empty","low","ok","full"} and confidence ∈ [0,1].
    The model is asked to classify by visual fill ratio:
      - empty: ~0-10% of bowl area contains kibbles
      - low:   ~10-30%
      - ok:    ~30-70%
      - full:  ~70-100%
    """
    # Downscale in-memory (no file save)
    max_side = 1024
    w, h = pil_img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    data_url = pil_to_data_url(pil_img)

    # Constrain output to a tiny JSON to be robust
    prompt_text = (
        "Classify the kibble level in the cat bowl by visual fill ratio.\n"
        "Use these buckets:\n"
        "- empty: ~0-10%\n"
        "- low:   ~10-30%\n"
        "- ok:    ~30-70%\n"
        "- full:  ~70-100%\n"
        'Respond ONLY with a compact JSON like: {"level":"ok","confidence":0.82}\n'
        "No extra text."
    )

    resp = _openai_client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {
                "role": "system",
                "content": "You are a vision assistant. Output only the required JSON. No explanations.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        temperature=0,
    )
    text = (resp.choices[0].message.content or "").strip()

    # Very defensive JSON parsing
    level = "ok"
    conf = 0.5
    try:
        import json

        j = json.loads(text)
        lv = str(j.get("level", "")).lower().strip()
        if lv in {"empty", "low", "ok", "full"}:
            level = lv
        c = float(j.get("confidence", 0.5))
        if 0.0 <= c <= 1.0:
            conf = c
    except Exception:
        # Fallback heuristic if model surprised us
        if "empty" in text.lower():
            level = "empty"
        elif "low" in text.lower():
            level = "low"
        elif "full" in text.lower():
            level = "full"
        else:
            level = "ok"
        conf = 0.5

    return level, conf


# ================== HISTORY ==================
def _ensure_history_header(path: str):
    """Create CSV with header if missing."""
    if not os.path.isfile(path):
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["timestamp_iso", "level", "confidence", "image_path"])
            print(f"[INFO] Created history CSV: {path}", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] Failed to create history CSV: {e}", file=sys.stderr)


def append_history(path: str, ts: datetime, level: str, conf: float, image_path: str):
    """Append one line to CSV history."""
    _ensure_history_header(path)
    try:
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([ts.isoformat(), level, f"{conf:.3f}", image_path or ""])
    except Exception as e:
        print(f"[WARN] Failed to append history: {e}", file=sys.stderr)


def read_history_since(path: str, since: datetime) -> list[dict]:
    """Read history entries since a given datetime."""
    if not os.path.isfile(path):
        return []
    out: list[dict] = []
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    ts = datetime.fromisoformat(row["timestamp_iso"])
                    if TZ and ts.tzinfo is None:
                        # If CSV had naive timestamps, localize roughly
                        ts = ts.replace(tzinfo=TZ)
                except Exception:
                    continue
                if ts >= since:
                    out.append(
                        {
                            "timestamp": ts,
                            "level": row.get("level", "ok"),
                            "confidence": float(row.get("confidence", "0.5") or 0.5),
                            "image_path": row.get("image_path", ""),
                        }
                    )
    except Exception as e:
        print(f"[WARN] Failed to read history: {e}", file=sys.stderr)
    return out


# ================== WEEKLY REPORT ==================
def _mark_weekly_report_sent(mark_path: str, week_id: str):
    """Store last week id to avoid duplicates."""
    try:
        with open(mark_path, "w", encoding="utf-8") as f:
            f.write(week_id)
    except Exception as e:
        print(f"[WARN] Failed to write weekly mark: {e}", file=sys.stderr)


def _read_weekly_report_mark(mark_path: str) -> str | None:
    if not os.path.isfile(mark_path):
        return None
    try:
        with open(mark_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None


def current_week_id(dt: datetime) -> str:
    """Return an identifier like '2025-W35' for the week of dt (ISO week)."""
    y, w, _ = dt.isocalendar()  # type: ignore
    return f"{y}-W{int(w):02d}"


def is_weekly_report_time(dt: datetime) -> bool:
    """Return True if dt matches the configured weekly day and time."""
    if dt.weekday() != WEEKLY_REPORT_DAY:
        return False
    return (dt.hour == WEEKLY_REPORT_HOUR) and (dt.minute >= WEEKLY_REPORT_MINUTE)


def build_weekly_report_text(dt: datetime) -> str:
    """Aggregate last 7 days and build a Telegram-friendly summary."""
    since = dt - timedelta(days=7)
    entries = read_history_since(HISTORY_PATH, since)
    if not entries:
        return "📊 Rapport hebdo: aucun enregistrement sur les 7 derniers jours."

    # Stats
    total = len(entries)
    counts = {"empty": 0, "low": 0, "ok": 0, "full": 0}
    for e in entries:
        counts[e["level"]] = counts.get(e["level"], 0) + 1
    alerts = counts["empty"] + counts["low"]

    # Find last state
    last = sorted(entries, key=lambda x: x["timestamp"])[-1]
    last_str = last["timestamp"].strftime("%Y-%m-%d %H:%M")

    # Simple streaks: last consecutive non-alerts from the end
    streak_ok = 0
    for e in reversed(sorted(entries, key=lambda x: x["timestamp"])):
        if e["level"] in {"ok", "full"}:
            streak_ok += 1
        else:
            break

    # Build text
    lines = [
        "📊 *Rapport hebdo croquettes* (7 derniers jours)",
        f"- Total vérifications: {total}",
        f"- Alertes (empty/low): {alerts} ({counts['empty']} empty, {counts['low']} low)",
        f"- Niveaux *ok/full*: {counts['ok']} ok, {counts['full']} full",
        f"- Dernier état ({last_str}): {last['level']}",
        f"- Streak sans alerte (fin de période): {streak_ok}",
    ]
    return "\n".join(lines)


# ================== CAPTURE + CLASSIFY ==================
def capture_and_classify(
    rtsp_url: str, analyze_after: float, use_ffmpeg: bool = True
) -> tuple[str, float, str]:
    """
    Open RTSP, wait N seconds, capture one frame, save it in last_images/,
    classify into {empty,low,ok,full}, then close.
    Returns (level, confidence, saved_image_path).
    """
    cap = open_capture(rtsp_url, use_ffmpeg=use_ffmpeg)
    if not cap.isOpened():
        raise RuntimeError("Failed to open RTSP stream")

    start_t = None
    frame = None
    try:
        while True:
            ok, f = cap.read()
            if not ok or f is None:
                raise RuntimeError("RTSP read failure")
            if start_t is None:
                start_t = time.time()
            if time.time() - start_t >= analyze_after:
                frame = f
                break

        pil_img = bgr_to_pil(frame)

        # Save and prune last 3 images for logs
        img_path = _save_and_prune_last_images(pil_img, out_dir="last_images", keep=3)

        # Vision
        level, conf = kibbles_level_with_gpt5_nano(pil_img)
        return level, conf, img_path
    finally:
        cap.release()


# ================== MAIN LOOP ==================
def main():
    parser = argparse.ArgumentParser(description="Kibbles counter service.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="One-shot test: capture a frame now and send a Telegram photo (ignores quiet hours).",
    )
    parser.add_argument(
        "--weekly-now",
        action="store_true",
        help="Send the weekly report right now (does not wait for scheduled time).",
    )
    args = parser.parse_args()

    rtsp_url = build_rtsp_url(
        ip=RTSP_TAPO_IP,
        port=RTSP_TAPO_PORT,
        user=RTSP_TAPO_USER,
        password=RTSP_TAPO_PASSWORD,
        stream=RTSP_TAPO_STREAM,
    )
    use_ffmpeg = not NO_FFMPEG

    print(
        f"[INFO] Service started. {rtsp_url} "
        f"interval={INTERVAL_SECONDS}s analyze_after={ANALYZE_AFTER}s "
        f"quiet_hours={QUIET_START_HOUR:02d}:00–{QUIET_END_HOUR:02d}:00 TZ={'system' if TZ is None else TZ} "
        f"weekly={WEEKLY_REPORT_DAY}@{WEEKLY_REPORT_HOUR:02d}:{WEEKLY_REPORT_MINUTE:02d}",
        file=sys.stderr,
    )

    # Initial RTSP connectivity check
    test_cap = open_capture(rtsp_url, use_ffmpeg=use_ffmpeg)
    if not test_cap.isOpened():
        err = "Could not open RTSP stream. Check URL or install ffmpeg: sudo apt install -y ffmpeg"
        print(f"[ERROR] {err}", file=sys.stderr)
        send_telegram_error(err)
        sys.exit(1)
    test_cap.release()

    # Ensure history exists
    _ensure_history_header(HISTORY_PATH)

    # ---------- TEST MODE ----------
    if args.test:
        try:
            level, conf, img_path = capture_and_classify(
                rtsp_url, ANALYZE_AFTER, use_ffmpeg=use_ffmpeg
            )
            ts = now_tz()
            append_history(HISTORY_PATH, ts, level, conf, img_path)
            print(
                f"[TEST] Forcing Telegram photo with level={level} conf={conf:.2f} path={img_path}",
                file=sys.stderr,
            )
            send_telegram_status(level, image_path=img_path)
            print("[TEST] Done.", file=sys.stderr)
            return
        except Exception as e:
            err_txt = f"TEST failed: {e}"
            print(f"[ERROR] {err_txt}", file=sys.stderr)
            send_telegram_error(err_txt)
            sys.exit(1)

    # ---------- WEEKLY NOW ----------
    if args.weekly_now:
        report = build_weekly_report_text(now_tz())
        send_telegram_message(report)
        # deliberately continue into loop

    # ---------- REGULAR LOOP ----------
    low_notified = False  # anti-spam latch for alerts
    error_notified = False  # anti-spam latch for errors

    # Weekly report dedup mark
    last_week_mark = _read_weekly_report_mark(LAST_REPORT_MARK)

    while True:
        # Respect quiet hours before doing anything
        now = now_tz()
        if in_quiet_hours(now):
            sleep_s = seconds_until_quiet_end(now)
            wake = now + timedelta(seconds=sleep_s)
            print(
                f"[INFO] Quiet hours active ({QUIET_START_HOUR:02d}:00–{QUIET_END_HOUR:02d}:00). "
                f"Sleeping until {wake.strftime('%Y-%m-%d %H:%M:%S %Z')}",
                file=sys.stderr,
            )
            time.sleep(sleep_s)
            continue

        loop_start = time.time()
        try:
            level, conf, img_path = capture_and_classify(
                rtsp_url, ANALYZE_AFTER, use_ffmpeg=use_ffmpeg
            )
            print(
                f"[INFO] Level={level} (conf={conf:.2f}, low_notified={low_notified})",
                file=sys.stderr,
            )

            # Append to history
            ts = now_tz()
            append_history(HISTORY_PATH, ts, level, conf, img_path)

            # Successful cycle clears error latch
            if error_notified:
                print("[INFO] Recovery: clearing error latch.", file=sys.stderr)
            error_notified = False

            # Notification policy: only alert on "low" or "empty"
            if level in {"empty", "low"}:
                if not low_notified:
                    send_telegram_status(level, image_path=img_path)
                    low_notified = True
            else:
                if low_notified:
                    print(
                        "[INFO] Level back to OK/FULL, reset notification latch.",
                        file=sys.stderr,
                    )
                low_notified = False

            # Weekly report check
            wk_id = current_week_id(now)
            if is_weekly_report_time(now) and last_week_mark != wk_id:
                report = build_weekly_report_text(now)
                send_telegram_message(report)
                _mark_weekly_report_sent(LAST_REPORT_MARK, wk_id)
                last_week_mark = wk_id
                print(f"[INFO] Weekly report sent for {wk_id}.", file=sys.stderr)

        except Exception as e:
            err_txt = str(e)
            print(f"[ERROR] {err_txt}", file=sys.stderr)
            if not error_notified:
                send_telegram_error(err_txt)  # one-time notification until recovery
                error_notified = True

        # Sleep until next run (unless we’re entering quiet hours during sleep)
        next_run_dt = now_tz() + timedelta(seconds=INTERVAL_SECONDS)
        print(
            f"[INFO] Next regular run scheduled at {next_run_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}",
            file=sys.stderr,
        )

        elapsed = time.time() - loop_start
        sleep_s = max(1, INTERVAL_SECONDS - int(elapsed))
        time.sleep(sleep_s)


if __name__ == "__main__":
    main()
