Prediction Workflow
===================

OpenKinetics Predictor runs prediction jobs asynchronously.

Workflow
--------

1. You upload a CSV file or submit JSON through the API.
2. The backend validates rows and method settings.
3. The backend creates a job record.
4. Celery queues the prediction task.
5. A worker runs the selected prediction engine.
6. The frontend polls job status.
7. You download results as CSV or JSON.

Progress tracking
-----------------

The service records job status during execution.

Status values include:

- Queued.
- Running.
- Completed.
- Failed.

For long jobs, progress updates show how many predictions finished.
