FROM python:3.12-slim

# The pipeline runs as a user without root access.
RUN useradd --create-home --uid 10001 runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY --chown=runner:runner main.py verify.py filers.csv /app/
COPY --chown=runner:runner agents/ /app/agents/
COPY --chown=runner:runner curator/ /app/curator/
COPY --chown=runner:runner scripts/ /app/scripts/

RUN mkdir -p /app/output /app/.cache && chown -R runner:runner /app/output /app/.cache

USER runner

ENTRYPOINT ["python", "/app/main.py"]
CMD ["--help"]
