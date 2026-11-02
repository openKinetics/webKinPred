from __future__ import annotations

import hmac
import json

from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from django.db import OperationalError
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from api.execution import control


@csrf_exempt
def executor(request, host_id, operation, attempt_id=None):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    configured = settings.EXECUTION_HOSTS.get(host_id, {})
    expected = configured.get("token", "")
    supplied = request.headers.get("Authorization", "")
    if not expected or not hmac.compare_digest(supplied, "Bearer " + expected):
        return JsonResponse({"error": "unauthorized"}, status=401)
    try:
        # Layered embeddings carry larger manifests than ordinary public JSON.
        # Read a bounded authenticated body instead of Django's public form limit.
        limit = 32 * 1024**2
        raw = request.read(limit + 1)
        if len(raw) > limit:
            return JsonResponse({"error": "execution manifest too large"}, status=413)
        data = json.loads(raw or b"{}")
        if not isinstance(data, dict):
            raise ValueError("Object body required")
        if operation.startswith("embed-"):
            from api.execution.embedding_api import dispatch

            return JsonResponse(dispatch(operation, data))
        if operation == "register":
            host = control.register(host_id, data)
            owned = list(
                host.attempts.filter(state__in=["leased", "running", "uncertain"]).values(
                    "id", "stop_requested"
                )
            )
            return JsonResponse(
                {
                    "paused": host.paused,
                    "budgets": host.budgets,
                    "owned_attempts": [row["id"] for row in owned],
                    "stop_attempts": [row["id"] for row in owned if row["stop_requested"]],
                }
            )
        if operation == "sequences":
            from api.execution.planning import sequence_id

            return JsonResponse({"ids": [sequence_id(seq) for seq in data["sequences"]]})
        if operation == "claim":
            return JsonResponse({"attempt": control.claim(host_id)})
        if operation == "collect-artifacts":
            from api.execution.retention import claim_cleanup

            return JsonResponse({"cleanup": claim_cleanup(host_id, version=data.get("version", 1))})
        if operation == "collected-artifacts":
            from api.execution.retention import finish_cleanup

            return JsonResponse(finish_cleanup(host_id, data))
        if operation == "reconcile":
            return JsonResponse(control.reconcile(host_id, data))
        if operation == "heartbeat" and attempt_id:
            return JsonResponse(control.heartbeat(host_id, attempt_id, data))
        if operation == "complete" and attempt_id:
            return JsonResponse(control.finish(host_id, attempt_id, data))
        if operation == "checkpoint" and attempt_id:
            return JsonResponse(control.checkpoint(host_id, attempt_id, data))
        if operation == "artifact-locations" and attempt_id:
            return JsonResponse(control.artifact_locations(host_id, attempt_id, data))
        if operation == "artifact-location" and attempt_id:
            return JsonResponse(control.artifact_location(host_id, attempt_id, data))
        if operation == "error" and attempt_id:
            return JsonResponse(control.fail(host_id, attempt_id, data))
        return JsonResponse({"error": "unknown operation"}, status=404)
    except PermissionError:
        return JsonResponse({"error": "ownership mismatch"}, status=403)
    except OperationalError as exc:
        if getattr(exc.__cause__, "pgcode", None) != "57014":
            raise
        return JsonResponse(
            {"error": "control temporarily busy"}, status=503, headers={"Retry-After": "5"}
        )
    except ObjectDoesNotExist:
        return JsonResponse({"error": "unknown execution record"}, status=404)
    except (ValueError, KeyError, TypeError) as exc:
        return JsonResponse({"error": str(exc)}, status=409)
