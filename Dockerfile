# syntax=docker/dockerfile:1.4
# Worker runtime image.
# Conda envs are expected to come from a prebuilt env image so regular deploys
# do not rebuild conda environments.

ARG WEBKINPRED_ENVS_IMAGE=webkinpred-envs:latest
FROM ${WEBKINPRED_ENVS_IMAGE}

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/conda/bin:$PATH"

# Django / app-level dependencies. Model-specific ML stacks are provided by the
# prebuilt conda environments in WEBKINPRED_ENVS_IMAGE.
COPY requirements.app.txt ./
RUN --mount=type=cache,id=webkinpred-pip-py313,target=/root/.cache/pip,sharing=locked \
    pip install -r requirements.app.txt

# Application code.
COPY . .

RUN mkdir -p /app/models/EITLEM/Weights \
             /app/models/CatPred \
             /app/models/TurNup/data/saved_models \
             /app/models/UniKP-main/models \
             /app/media/sequence_info \
             /app/staticfiles \
             /app/mmseqs_tmp \
    && chmod 777 /app/mmseqs_tmp

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health/ || exit 1

CMD ["celery", "-A", "webKinPred", "worker", \
     "--loglevel=info", "--queues=webkinpred", "--concurrency=1"]
