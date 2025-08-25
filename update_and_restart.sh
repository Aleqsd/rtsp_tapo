#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

NAME="croquettes"
APP_DIR="$HOME/rtsp_tapo"
LOG="$APP_DIR/croquettes.log"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG" ; }

log "==== update_and_restart.sh invoked ===="

# Kill screen if running
if screen -ls | grep -q "\.${NAME}\s"; then
  log "Stopping existing screen session '${NAME}'..."
  screen -S "${NAME}" -X quit || log "WARN: failed to kill existing session."
else
  log "No existing screen session '${NAME}' found."
fi

# Update repo
log "Running git pull..."
if git -C "$APP_DIR" pull --rebase --autostash; then
  log "Repo updated successfully."
else
  log "ERROR: git pull failed."
  exit 1
fi

# Restart program
log "Restarting screen session '${NAME}'..."
"$APP_DIR/start_screen.sh"

log "==== update_and_restart.sh done ===="
