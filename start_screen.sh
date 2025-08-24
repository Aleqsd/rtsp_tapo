#!/usr/bin/env bash
set -euo pipefail

NAME="croquettes"
APP_DIR="$HOME/rtsp_tapo"

# Wait a bit for the network to be ready after boot
sleep 15

# Ensure screen is installed
command -v screen >/dev/null 2>&1 || sudo apt update && sudo apt install -y screen

# Kill any existing session with the same name (to avoid duplicates)
if screen -ls | grep -q "\.${NAME}\s"; then
  screen -S "${NAME}" -X quit || true
fi

# Start a detached screen session running the loop script
# -dmS : detached mode + session name
# bash -lc : login shell (ensures ~/.profile or ~/.bashrc is sourced)
screen -dmS "${NAME}" bash -lc "cd '${APP_DIR}' && ./run_loop.sh"
