import sys
import time
import os
import re
import base64
import cv2
import requests
from PIL import Image
import numpy as np
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
INTERVAL_SECONDS = int(
    os.getenv("INTERVAL_SECONDS", "3600")
)  # interval between runs (default 1h)
THRESHOLD = int(os.getenv("THRESHOLD", "30"))  # notify if < THRESHOLD
NO_FFMPEG = os.getenv("NO_FFMPEG", "0").lower() in ("1", "true", "yes")

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
def send_telegram_count(count: int):
    """Send a Telegram notification with the kibble count."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    msg = f"Il reste {count} croquettes 🐾"
    try:
        r = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": msg,
                "disable_notification": False,
            },
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
        max_tokens=8,
        temperature=0,
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
) -> int:
    """Open RTSP, wait N seconds, capture one frame, count kibbles, then close."""
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
        count = count_kibbles_with_gpt5_nano(pil_img)
        return count
    finally:
        cap.release()


# ================== MAIN LOOP ==================
def main():
    rtsp_url = build_rtsp_url(
        ip=RTSP_TAPO_IP,
        port=RTSP_TAPO_PORT,
        user=RTSP_TAPO_USER,
        password=RTSP_TAPO_PASSWORD,
        stream=RTSP_TAPO_STREAM,
    )
    use_ffmpeg = not NO_FFMPEG

    print(
        f"[INFO] Service started. RTSP={rtsp_url} "
        f"interval={INTERVAL_SECONDS}s threshold<{THRESHOLD} analyze_after={ANALYZE_AFTER}s",
        file=sys.stderr,
    )

    # ✅ Initial RTSP connectivity check
    test_cap = open_capture(rtsp_url, use_ffmpeg=use_ffmpeg)
    if not test_cap.isOpened():
        err = "Could not open RTSP stream. Check URL or install ffmpeg: sudo apt install -y ffmpeg"
        print(f"[ERROR] {err}", file=sys.stderr)
        send_telegram_error(err)  # one-time alert at startup if stream fails
        sys.exit(1)
    test_cap.release()

    low_notified = False  # anti-spam latch for threshold
    error_notified = False  # anti-spam latch for errors

    while True:
        loop_start = time.time()
        try:
            count = capture_and_count(rtsp_url, ANALYZE_AFTER, use_ffmpeg=use_ffmpeg)
            print(count)  # stdout for logging/metrics
            print(
                f"[INFO] Count={count} (threshold={THRESHOLD}, low_notified={low_notified})",
                file=sys.stderr,
            )

            # ✅ Successful cycle resets error latch
            if error_notified:
                print("[INFO] Recovery: clearing error latch.", file=sys.stderr)
            error_notified = False

            # Threshold notification (anti-spam)
            if count < THRESHOLD:
                if not low_notified:
                    send_telegram_count(count)
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

        # Sleep until next run (accounting for elapsed time)
        elapsed = time.time() - loop_start
        sleep_s = max(1, INTERVAL_SECONDS - int(elapsed))
        time.sleep(sleep_s)


if __name__ == "__main__":
    main()
