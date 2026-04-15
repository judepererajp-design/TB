#!/bin/bash
# TitanBot Pro — Start Script
# Includes an auto-restart loop so a crash doesn't leave the bot permanently down.

set -e

echo "🤖 Starting TitanBot Pro..."

# Check .env
if [ ! -f .env ]; then
    echo "❌ .env file not found. Copy .env.example to .env and fill in your credentials."
    exit 1
fi

# Export env vars
set -a
source .env
set +a

# Create logs directory
mkdir -p logs

# D5-FIX: Auto-restart loop — if the bot crashes it restarts after a brief
# backoff rather than staying down until manually restarted.
# Press Ctrl+C (SIGINT) twice quickly to exit completely.
MAX_RESTARTS=20          # safety cap — avoids infinite crash loops
RESTART_DELAY=10         # seconds to wait before restarting
restart_count=0

while true; do
    echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') ▶ Starting TitanBot Pro (attempt $((restart_count + 1)))..."
    python3 main.py
    EXIT_CODE=$?

    # Exit code 0 means clean shutdown (Ctrl+C / SIGTERM) — do not restart
    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') ✅ TitanBot stopped cleanly."
        break
    fi

    restart_count=$((restart_count + 1))
    if [ "$restart_count" -ge "$MAX_RESTARTS" ]; then
        echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') ❌ Reached max restarts ($MAX_RESTARTS). Giving up."
        exit 1
    fi

    echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') ⚠️  Bot exited with code $EXIT_CODE. Restarting in ${RESTART_DELAY}s... (${restart_count}/${MAX_RESTARTS})"
    sleep "$RESTART_DELAY"
done
