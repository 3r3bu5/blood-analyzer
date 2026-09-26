# WBC Blood Film Analyzer — reproducible runtime for the research POC.
# Base image stays lean (no ML weights or datasets baked in). Mount data,
# models, and outputs at run time; see compose.yaml.
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS runtime

# Set to 1 at build time to include the heavy ML extras (torch, ultralytics, ...).
ARG INSTALL_ML=0

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/
COPY scripts/ scripts/

RUN pip install --no-cache-dir . \
    && if [ "$INSTALL_ML" = "1" ]; then pip install --no-cache-dir ".[ml]"; fi

COPY configs/ configs/

ENTRYPOINT ["bloodfilm"]
CMD ["--help"]

# Test stage: lean runtime plus the dev extras so the suite runs in-container.
FROM runtime AS test

RUN pip install --no-cache-dir ".[dev]"

COPY tests/ tests/

ENTRYPOINT ["python", "-m", "pytest", "-q"]
CMD []
