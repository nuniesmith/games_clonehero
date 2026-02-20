#!/bin/sh
set -e

# Stateless container entrypoint.
# No local data directories are needed — all persistent storage is on Nextcloud.
# The only job here is to drop privileges from root to appuser before
# executing the main process (Gunicorn / Uvicorn).

exec su -s /bin/sh appuser -c "exec $*" -- "$@"
