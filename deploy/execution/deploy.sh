#!/usr/bin/env bash
# Deploy already built, pinned images; retain legacy queues and rollback images.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${EXECUTION_ENV_FILE:?Set EXECUTION_ENV_FILE to the private release environment file}"
if [[ ! -f /run/webkinpred/execution-v2 ]]; then
    echo 'Complete the frozen-queue migration before using routine execution deployments.' >&2
    exit 1
fi
legacy=$(sudo docker ps --filter label=com.docker.compose.project=webkinpred \
    --format '[{{json (.Label "com.docker.compose.service")}},{{json (.Label "org.webkinpred.profiling")}},{{json (.Label "org.webkinpred.executor")}}]' \
    | python3 -c 'import json,sys; print("\n".join(service for service,probe,executor in map(json.loads,sys.stdin) if service in {"celery","celery-cache","celery-beat"} and probe != "true" and not executor))')
if [[ -n "$legacy" ]]; then
    echo 'Legacy consumers are still running; follow the cutover procedure first.' >&2
    exit 1
fi
compose=(sudo docker compose -p webkinpred -f docker-compose.prod.yml -f docker-compose.execution.yml --env-file "$EXECUTION_ENV_FILE")
if [[ -n "${EXECUTION_COMPOSE_OVERRIDE:-}" ]]; then
    compose+=(-f "$EXECUTION_COMPOSE_OVERRIDE")
fi
# The migration service runs once before web/control processes are replaced.
"${compose[@]}" up -d --no-build execution-db
"${compose[@]}" --profile migration run --rm --no-deps execution-migrate
if (( $# )); then
    for service in "$@"; do
        case "$service" in celery|celery-cache|celery-beat)
            echo 'Legacy consumers are disabled after migration.' >&2
            exit 1
        esac
    done
    "${compose[@]}" up -d --no-build "$@"
else
    "${compose[@]}" up -d --no-build backend frontend execution-controller execution-control-worker execution-agent execution-artifacts
fi
# Never prune rollback images or remove unrelated/legacy containers here.

# nginx resolves its upstream when configuration loads. Refresh it after a
# backend container replacement even when the frontend image is unchanged.
frontend_id=$("${compose[@]}" ps -q frontend)
if [[ -n "$frontend_id" ]] && [[ $(sudo docker inspect --format '{{.State.Running}}' "$frontend_id") == true ]]; then
    sudo docker exec "$frontend_id" nginx -t
    sudo docker exec "$frontend_id" nginx -s reload
fi
