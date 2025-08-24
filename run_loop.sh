#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Load environment variables from ~/.bashrc if needed
# Uncomment if your API keys are only in ~/.bashrc:
# source "$HOME/.bashrc"

# Sanity checks to ensure required env vars are present
: "${OPEN_AI_API_KEY:?Missing OPEN_AI_API_KEY}"
: "${RTSP_TAPO_USER:?Missing RTSP_TAPO_USER}"
: "${RTSP_TAPO_PASSWORD:?Missing RTSP_TAPO_PASSWORD}"
: "${RTSP_TAPO_IP:?Missing RTSP_TAPO_IP}"
: "${TELEGRAM_BOT_TOKEN:?Missing TELEGRAM_BOT_TOKEN}"
: "${TELEGRAM_CHAT_ID:?Missing TELEGRAM_CHAT_ID}"

# Infinite loop to auto-restart the bot if it crashes
while true; do
  date -Iseconds >> croquettes.err.log
  python3 main.py >> croquettes.out.log 2>> croquettes.err.log || true
  echo "crash/restart in 10s" >> croquettes.err.log
  sleep 10
done
