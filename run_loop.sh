#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

LOG_FILE="croquettes.log"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG_FILE" ; }

log "==== run_loop.sh invoked ===="

# Infinite loop to auto-restart main.py
while true; do
  log "Launching main.py…"
  # Append both stdout and stderr from main.py to the same log file
  if python3 main.py >> "$LOG_FILE" 2>&1; then
    log "main.py exited normally (unexpected, restarting in 10s)."
  else
    log "main.py crashed or returned non-zero, restarting in 10s."
  fi
  sleep 10
done
