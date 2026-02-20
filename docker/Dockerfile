###############################
# Stage 1: Builder
###############################
FROM python:3.13-slim AS builder

WORKDIR /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libsndfile1-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a Python virtual environment
RUN python -m venv /opt/venv

# Install Python dependencies into the virtual environment
COPY requirements.txt /app/requirements.txt
RUN /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

###############################
# Stage 2: Runtime
###############################
FROM python:3.13-slim

# Set runtime environment variables
# No DATA_DIR or LOG_DIR — the container is stateless.
# The SQLite DB is downloaded from Nextcloud on startup and lives in /tmp.
# Logs go to stdout only.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app" \
    PATH="/opt/venv/bin:$PATH" \
    APP_HOST="0.0.0.0" \
    APP_PORT="8000"

WORKDIR /app

# Install only necessary runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libffi8 \
    curl \
    unrar-free \
    p7zip-full \
    fonts-dejavu-core \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application source code
COPY src/ /app/src/

# Copy the entrypoint script
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create a dedicated non-root user
RUN useradd -m -r appuser && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app

# Expose the application port
EXPOSE 8000

# Health Check: Ensure the service is responding
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8000/api/health || exit 1

# Entrypoint drops privileges from root to appuser
ENTRYPOINT ["/app/entrypoint.sh"]

# Start with Gunicorn + Uvicorn workers (production)
# For development, override with: uvicorn src.main:app --reload
CMD ["gunicorn", \
    "-w", "1", \
    "-k", "uvicorn.workers.UvicornWorker", \
    "--bind", "0.0.0.0:8000", \
    "--timeout", "300", \
    "--graceful-timeout", "30", \
    "--access-logfile", "-", \
    "--error-logfile", "-", \
    "src.main:app"]
