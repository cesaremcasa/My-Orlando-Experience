FROM python:3.11-slim-bookworm

RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir --disable-pip-version-check ".[agentops]" httpx==0.28.1 \
    && rm -rf /root/.cache/pip

USER 10001
ENV PYTHONUNBUFFERED=1
ENV ORLANDO_FAKE_RUNTIME=0
ENV ORLANDO_AGENTOPS_DATA_DIR=/tmp/agentops
EXPOSE 8080

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
