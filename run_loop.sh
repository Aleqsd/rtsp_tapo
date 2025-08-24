#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

LOG_ERR="croquettes.err.log"
LOG_OUT="croquettes.out.log"
ENV_FILE="$HOME/.croquettes_env"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG_ERR" ; }

log "==== run_loop.sh invoked ===="

# Load environment variables if available
if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  log "Environment file $ENV_FILE loaded."
else
  log "WARN: environment file $ENV_FILE not found, relying on inherited env."
fi

# Infinite loop to auto-restart main.py
while true; do
  log "Launching main.py…"
  if python3 main.py >> "$LOG_OUT" 2>> "$LOG_ERR"; then
    log "main.py exited normally (unexpected, restarting in 10s)."
  else
    log "main.py crashed or returned non-zero, restarting in 10s."
  fi
  sleep 10
done
