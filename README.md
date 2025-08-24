# 🐾 Croquettes Counter Bot

This project uses a Tapo RTSP camera + OpenAI Vision API to **estimate the number of kibbles in a bowl**.  
It notifies you via Telegram when the number of kibbles drops below a threshold.

Runs on a Freebox Delta VM (Debian) or any Linux machine.

---

## 📦 Features

- Capture one frame from the RTSP feed every `N` seconds (default: 1h).
- Send the frame to **OpenAI GPT-5 Nano (vision)** to count kibbles.
- If count < threshold (default: 30), send a **Telegram notification**.
- Anti-spam: only notify once until count goes back above threshold.
- Robust logging (all logs in `croquettes.log`).
- Resilient: auto-restart loop (`run_loop.sh`) and relaunch on reboot (`cron + screen`).

---

## ⚙️ Requirements

- Debian / Ubuntu machine (Freebox Delta VM works fine).
- Packages:

  ```bash
  sudo apt update
  sudo apt install -y python3 python3-pip ffmpeg screen cron git
  ```

- Python packages:
  ```bash
  pip3 install --break-system-packages openai opencv-python pillow requests numpy
  ```

---

## 🔑 Environment Variables

The bot uses environment variables for configuration.  
Add them to `~/.bashrc` or `~/.profile`:

```bash
export OPENAI_API_KEY="sk-xxx"
export RTSP_TAPO_USER="admin"
export RTSP_TAPO_PASSWORD="password"
export RTSP_TAPO_IP="192.168.1.45"
export TELEGRAM_BOT_TOKEN="123456:ABC-DEF"
export TELEGRAM_CHAT_ID="123456789"
```

Reload your shell:

```bash
source ~/.bashrc
```

---

## 🚀 Usage

### 1. Clone the repo

```bash
git clone https://github.com/your-username/rtsp_tapo.git
cd rtsp_tapo
chmod +x start_screen.sh run_loop.sh
```

### 2. Run manually

```bash
./start_screen.sh
```

Check running sessions:

```bash
screen -ls
```

Attach to session:

```bash
screen -r croquettes
# Detach with Ctrl+A then D
```

### 3. Logs

All logs go to:

```bash
tail -f croquettes.log
```

---

## 🔄 Auto-start on reboot

Enable `cron` (if not already):

```bash
sudo systemctl enable cron
sudo systemctl start cron
```

Edit crontab:

```bash
crontab -e
```

Add:

```
@reboot /home/freebox/rtsp_tapo/start_screen.sh
```

Now the bot will relaunch automatically after every reboot.

---

## 🧪 Testing

- To simulate low kibble count, temporarily lower the `--threshold` value in `main.py` or edit your environment variables.
- You can check Telegram logs in `croquettes.log` to confirm notifications were sent.

---

## 📂 Project Structure

```
rtsp_tapo/
├── main.py             # Main Python script (AI + RTSP capture + Telegram notify)
├── run_loop.sh         # Keeps main.py running, auto-restarts on crash
├── start_screen.sh     # Manages screen session, auto-started via cron
├── croquettes.log      # Unified log file (created at runtime)
└── README.md           # This file
```

---

## 🛠 Troubleshooting

- `screen -ls` shows no session → check `croquettes.log`.
- `ModuleNotFoundError: openai` → ensure `pip3 install openai --break-system-packages`.
- No Telegram messages → double-check `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.
- Camera errors → verify RTSP URL:
  ```
  rtsp://<user>:<password>@<ip>:554/stream1
  ```

---

## ✨ Notes

- Uses **GPT-5 Nano (Vision)** for lowest cost (~$0.00003 per request).
- Designed to be lightweight and always-on.
- Easy to extend: you can adapt `main.py` to detect other objects, not just kibbles 🐕🐈.
