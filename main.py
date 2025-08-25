import sys
import time
import os
import re
import glob
import base64
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
THRESHOLD = int(os.getenv("THRESHOLD", "30"))  # notify if < THRESHOLD
NO_FFMPEG = os.getenv("NO_FFMPEG", "0").lower() in ("1", "true", "yes")

# Quiet hours window (inclusive start, exclusive end)
QUIET_START_HOUR = int(os.getenv("QUIET_START_HOUR", "22"))  # 22:00
QUIET_END_HOUR = int(os.getenv("QUIET_END_HOUR", "8"))  # 08:00

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


def send_telegram_count(count: int, image_path: str | None = None):
    """Send a Telegram notification with kibble count, including photo if available."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(
            "[WARN] Telegram env vars not set; skipping notification.", file=sys.stderr
        )
        return

    caption = f"Il reste {count} croquettes 🐾"
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
        print(
            "[INFO] Telegram notification (with photo)"
            if image_path
            else "[INFO] Telegram notification sent.",
            file=sys.stderr,
        )
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


def count_kibbles_with_gpt5_nano(pil_img: Image.Image) -> int:
    """Send the image to GPT-5 nano and return a single integer."""
    # Downscale in-memory (no file save)
    max_side = 1024
    w, h = pil_img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    data_url = pil_to_data_url(pil_img)

    prompt_text = (
        "Count the number of individual kibbles in this bowl. "
        "If some overlap, estimate as best as possible. "
        "Respond with a single integer only. No words, no units."
        "You will only respond with text if there's an error."
    )

    resp = _openai_client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {
                "role": "system",
                "content": "You are an object counter. Output only a single integer with no words.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("Empty response from model")
    m = re.search(r"-?\d+", text)
    if not m:
        raise RuntimeError(f"Model did not return an integer: {text!r}")
    return int(m.group(0))


def capture_and_count(
    rtsp_url: str, analyze_after: float, use_ffmpeg: bool = True
) -> tuple[int, str]:
    """
    Open RTSP, wait N seconds, capture one frame, save it in last_images/,
    count kibbles, then close. Returns (count, saved_image_path).
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

        count = count_kibbles_with_gpt5_nano(pil_img)
        return count, img_path
    finally:
        cap.release()


# ================== MAIN LOOP ==================
def main():
    parser = argparse.ArgumentParser(description="Kibbles counter service.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="One-shot test: capture a frame now and send a Telegram photo (ignores quiet hours and threshold).",
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
        f"interval={INTERVAL_SECONDS}s threshold<{THRESHOLD} analyze_after={ANALYZE_AFTER}s "
        f"quiet_hours={QUIET_START_HOUR:02d}:00–{QUIET_END_HOUR:02d}:00 TZ={'system' if TZ is None else TZ}",
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

    # ---------- TEST MODE ----------
    if args.test:
        try:
            count, img_path = capture_and_count(
                rtsp_url, ANALYZE_AFTER, use_ffmpeg=use_ffmpeg
            )
            print(
                f"[TEST] Forcing Telegram photo with count={count} path={img_path}",
                file=sys.stderr,
            )
            send_telegram_count(count, image_path=img_path)
            print("[TEST] Done.", file=sys.stderr)
            return
        except Exception as e:
            err_txt = f"TEST failed: {e}"
            print(f"[ERROR] {err_txt}", file=sys.stderr)
            send_telegram_error(err_txt)
            sys.exit(1)

    # ---------- REGULAR LOOP ----------
    low_notified = False  # anti-spam latch for threshold
    error_notified = False  # anti-spam latch for errors

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
            count, img_path = capture_and_count(
                rtsp_url, ANALYZE_AFTER, use_ffmpeg=use_ffmpeg
            )
            print(
                f"[INFO] Count={count} (threshold={THRESHOLD}, low_notified={low_notified})",
                file=sys.stderr,
            )

            # Successful cycle clears error latch
            if error_notified:
                print("[INFO] Recovery: clearing error latch.", file=sys.stderr)
            error_notified = False

            # Threshold notification (anti-spam), include photo
            if count < THRESHOLD:
                if not low_notified:
                    send_telegram_count(count, image_path=img_path)
                    low_notified = True
            else:
                if low_notified:
                    print(
                        "[INFO] Count back above threshold, reset notification latch.",
                        file=sys.stderr,
                    )
                low_notified = False

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

        # If the next run would fall inside quiet hours, we’ll handle it at loop start.
        elapsed = time.time() - loop_start
        sleep_s = max(1, INTERVAL_SECONDS - int(elapsed))
        time.sleep(sleep_s)


if __name__ == "__main__":
    main()
