Logging and Observability
=========================

webKinPred writes application logs as one JSON object per line. Docker
still prefixes lines in ``docker compose logs`` (for example
``celery-1 |``), but the application payload after that prefix is JSON
and can be parsed by Loki, ``jq``, or other log tooling.

Log Shape
---------

Common fields:

- ``timestamp``: UTC ISO timestamp.
- ``level``: Python log level.
- ``service``: ``backend``, ``celery``, ``celery-beat``, or
  ``webkinpred`` fallback.
- ``logger``: Python logger name.
- ``event``: stable event key, such as ``subprocess.started``.
- ``request_id``: propagated from ``X-Request-ID`` or generated per
  Django request.
- ``job_public_id``: public job id, when known.
- ``celery_task_id``: Celery task id, when known.
- ``method_key``: prediction method, when known.
- ``target``: prediction target, when known.

Useful Celery events:

- ``celery.task.started``
- ``celery.task.finished``
- ``celery.task.failed``
- ``prediction.method_started``
- ``subprocess.started``
- ``subprocess.progress``
- ``subprocess.stderr_line``
- ``subprocess.completed``
- ``subprocess.failed``
- ``http.request``
- ``http.request_failed``

The prediction subprocess protocol is unchanged: method scripts may
still write ``Progress: x/y`` to stdout. The backend parses those lines
for database-backed frontend progress. Operational logs are separate and
structured. Gunicorn access logs are disabled in Docker because the
Django request middleware emits structured request logs instead.

Local Development
-----------------

Run the normal development stack:

.. code:: bash

   docker compose up backend celery celery-beat redis frontend

Inspect structured Celery logs:

.. code:: bash

   docker compose logs celery --tail 100

Increase subprocess verbosity when debugging model output:

.. code:: bash

   SUBPROCESS_LOG_LEVEL=DEBUG docker compose up celery

Production Observability Stack
------------------------------

The production compose file includes Loki, Promtail, and Grafana.

.. code:: bash

   docker compose -f docker-compose.prod.yml up -d loki promtail grafana

Grafana is exposed on port ``3001`` by default. Set credentials with:

.. code:: bash

   GRAFANA_ADMIN_USER=admin GRAFANA_ADMIN_PASSWORD='change-me' docker compose -f docker-compose.prod.yml up -d grafana

Promtail reads Docker container logs, parses the JSON payload, and
labels logs with ``service``, ``level``, ``event``, ``job_public_id``,
``celery_task_id``, ``method_key``, and ``target``.

If your host Docker daemon rejects Promtail Docker API discovery (for
example API version mismatch), webKinPred uses file-based scraping from
``/var/lib/docker/containers/*/*-json.log``.

After updating Promtail config, force recreate the Promtail container so
it remounts the latest host file:

.. code:: bash

   docker compose -f docker-compose.prod.yml rm -sf promtail
   docker compose -f docker-compose.prod.yml up -d --no-deps --force-recreate promtail

Then verify the container sees the updated config:

.. code:: bash

   docker compose -f docker-compose.prod.yml exec -T promtail \
     sh -lc 'grep -n "job_name" /etc/promtail/config.yml && grep -n "docker_sd_configs" /etc/promtail/config.yml || true'

Expected:

- ``job_name: docker-file-logs``
- no ``docker_sd_configs`` block

Common LogQL Queries
--------------------

All logs for a job:

.. code:: text

   {service="celery"} | json | job_public_id="mReUVED"

All logs for a Celery task:

.. code:: text

   {service="celery"} | json | celery_task_id="f4007945-92fe-41c0-9229-127304d14e9d"

Method failures:

.. code:: text

   {service="celery"} | json | level="ERROR" | method_key="EITLEM"

Subprocess output warnings/errors:

.. code:: text

   {service="celery", event="subprocess.stderr_line"} | json

Completed subprocess durations:

.. code:: text

   {service="celery", event="subprocess.completed"} | json | unwrap duration_ms

Backend request errors:

.. code:: text

   {service="backend"} | json | level="ERROR"

Contributor Rules
-----------------

- Do not add bare ``print()`` for operational logs in ``api/`` runtime
  code.
- Use ``logging.getLogger(__name__)`` and add stable ``event`` names in
  ``extra``.
- Keep user-facing SSE/session messages on ``push_line()``; those are
  product UX, not infrastructure logs.
- Keep method subprocess progress lines as ``Progress: x/y`` when adding
  new prediction scripts.
