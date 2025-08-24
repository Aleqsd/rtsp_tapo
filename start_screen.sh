#!/usr/bin/env bash
set -euo pipefail

NAME="croquettes"
APP_DIR="$HOME/rtsp_tapo"
LOG="$APP_DIR/croquettes.log"

# --- logging helper ---
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG" ; }

log "==== start_screen.sh invoked ===="

# Wait a bit for the network to be ready after boot
log "Sleeping 15s to wait for network…"
sleep 15

# Minimal context dump
log "USER=$(id -un) HOME=$HOME SHELL=$SHELL"
log "APP_DIR=$APP_DIR"

# Ensure screen is installed
if ! command -v screen >/dev/null 2>&1; then
  log "screen not found → installing…"
  if sudo apt update && sudo apt install -y screen; then
    log "screen installed."
  else
    log "ERROR: failed to install screen."
    exit 1
  fi
else
  log "screen already installed."
fi

# Check app dir exists
if [ ! -d "$APP_DIR" ]; then
  log "ERROR: APP_DIR '$APP_DIR' does not exist."
  exit 1
fi

# Quick env sanity (mask secrets in logs)
mask() { sed -E 's/./*/g' ; }
log "ENV sanity: OPEN_AI_API_KEY=${OPEN_AI_API_KEY:+$(echo "$OPEN_AI_API_KEY" | mask)}"
log "ENV sanity: RTSP_TAPO_USER=${RTSP_TAPO_USER:-<unset>} RTSP_TAPO_IP=${RTSP_TAPO_IP:-<unset>}"
log "ENV sanity: TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID:-<unset>} TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN:+set}"

# Check loop script
if [ ! -x "$APP_DIR/run_loop.sh" ]; then
  log "Making run_loop.sh executable."
  chmod +x "$APP_DIR/run_loop.sh" || { log "ERROR: chmod run_loop.sh failed."; exit 1; }
fi

# Kill any existing session with the same name (to avoid duplicates)
if screen -ls | grep -q "\.${NAME}\s"; then
  log "Existing screen session '${NAME}' found → killing it."
  screen -S "${NAME}" -X quit || log "WARN: failed to kill existing session (may be fine)."
else
  log "No existing screen session '${NAME}'."
fi

# Start a detached screen session running the loop script
CMD="cd '$APP_DIR' && ./run_loop.sh"
log "Starting screen session '${NAME}' with command: $CMD"
if screen -dmS "${NAME}" bash -lc "$CMD"; then
  log "screen session '${NAME}' started (detached)."
else
  log "ERROR: failed to start screen session."
  exit 1
fi

# Verify it’s really running
sleep 1
if screen -ls | grep -q "\.${NAME}\s"; then
  log "OK: screen session '${NAME}' is listed by screen -ls."
else
  log "ERROR: screen session '${NAME}' not found after start."
  log "Hint: check $LOG for runtime errors."
  exit 1
fi

log "==== start_screen.sh done ===="
