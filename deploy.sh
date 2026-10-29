#!/usr/bin/env bash
# deploy.sh — rebuild changed services and reclaim space from old image layers
#
# Usage:
#   ./deploy.sh                     # prod (default, prunes dangling images)
#   ./deploy.sh dev                 # dev compose
#   ./deploy.sh prod celery         # rebuild only the celery worker
#   SKIP_PRUNE=1 ./deploy.sh        # skip dangling image prune
#
# How space is reclaimed:
#   Each `--build` tags the new image (e.g. webkinpred-worker:latest) and
#   leaves the old image untagged ("dangling"). Docker never auto-removes
#   these. `docker image prune -f` at the end cleans them up.
#
# How the cache is preserved:
#   BuildKit reuses unchanged layers and pip download caches.
#   Conda envs are now expected to come from a prebuilt env image
#   (WEBKINPRED_ENVS_IMAGE), so regular worker deploys avoid rebuilding them.

set -euo pipefail

ENV="${1:-prod}"
shift 1 2>/dev/null || true   # remaining args = specific services to rebuild

if [[ "$ENV" == "prod" && ( -n "${EXECUTION_ENV_FILE:-}" || -f /run/webkinpred/execution-v2 ) ]]; then
  if [[ -z "${EXECUTION_ENV_FILE:-}" ]]; then
    echo 'Resource-aware execution is installed. Set EXECUTION_ENV_FILE to its private release environment.' >&2
    exit 1
  fi
  exec ./deploy/execution/deploy.sh "$@"
fi

case "$ENV" in
  prod)
    COMPOSE_FILE="docker-compose.prod.yml"
    ENV_FILE=".env.production"
    ;;
  dev)
    COMPOSE_FILE="docker-compose.yml"
    ENV_FILE=".env"
    ;;
  *)    echo "Unknown env '$ENV'. Use 'prod' or 'dev'."; exit 1 ;;
esac

COMPOSE_ARGS=(-f "$COMPOSE_FILE")
if [[ -f "$ENV_FILE" ]]; then
  COMPOSE_ARGS+=(--env-file "$ENV_FILE")
  echo "==> Using env file: $ENV_FILE"
elif [[ "$ENV" == "prod" && -f ".env" ]]; then
  COMPOSE_ARGS+=(--env-file ".env")
  echo "==> Using env file: .env (fallback)"
else
  echo "==> No env file found for '$ENV' (expected $ENV_FILE). Using current shell env."
fi

echo "==> Building and starting services (compose: $COMPOSE_FILE) ..."
DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 \
  sudo docker compose "${COMPOSE_ARGS[@]}" up -d --build --remove-orphans "$@"

if [[ "${SKIP_PRUNE:-0}" != "1" ]]; then
  echo "==> Pruning dangling images to reclaim disk space ..."
  sudo docker image prune -f
else
  echo "==> Skipping dangling image prune (SKIP_PRUNE=1)."
fi

echo "==> Done."
