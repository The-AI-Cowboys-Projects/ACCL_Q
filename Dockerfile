# ACCL-Q - IBM Cloud Code Engine Deployment
# ==========================================
#
# Deploys ACCL-Q quantum emulator as a serverless container.
# Provides REST API for quantum collective operations simulation.

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    numpy>=1.20.0 \
    fastapi>=0.109.0 \
    uvicorn[standard]>=0.27.0 \
    pydantic>=2.0.0

# Copy ACCL-Q driver
COPY driver/python/accl_quantum /app/accl_quantum

# Copy API server
COPY api_server.py /app/

# Environment
ENV PYTHONUNBUFFERED=1 \
    PORT=8080

# Create non-root user
RUN groupadd -r accl && useradd -r -g accl accl
USER accl

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["python", "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8080"]
