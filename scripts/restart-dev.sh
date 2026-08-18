#!/usr/bin/env bash
# Restart the dev server on a fixed port, freeing it first.
#
# Output is unbuffered and tee'd to dev-server.log. Without -u, Python buffers
# stdout when it is a pipe, so tracebacks sit in the buffer instead of reaching
# the log -- which makes a crashing handler look like silence.
set -u
PORT="${1:-8080}"
LOG="${REDLINER_LOG:-dev-server.log}"

PID=$(netstat -ano | grep ":${PORT}.*LISTENING" | awk '{print $5}' | head -1)
if [ -n "${PID:-}" ]; then
  taskkill //F //PID "$PID" >/dev/null 2>&1
  sleep 2
fi

exec ./.venv/Scripts/python.exe -u -m redliner --port "$PORT" 2>&1 | tee "$LOG"
