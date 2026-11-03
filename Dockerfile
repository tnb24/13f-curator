FROM python:3.12-slim

# The pipeline and the agent run as a user without root access.
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
COPY --chown=runner:runner evals/ /app/evals/
COPY --chown=runner:runner scripts/ /app/scripts/
COPY --chown=runner:runner output/ /app/output/

RUN mkdir -p /app/.cache && chown -R runner:runner /app/output /app/.cache

USER runner

# Build the dataset:   docker run IMAGE --user-agent "Name you@example.com"
# Ask the agent:       docker run --env-file .env --entrypoint python IMAGE -m agents.answer "..."
ENTRYPOINT ["python", "/app/main.py"]
CMD ["--help"]
