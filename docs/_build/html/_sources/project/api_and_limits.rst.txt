API and Limits
==============

OpenKinetics Predictor provides a REST API for programmatic access.

Base URL
--------

https://predictor.openkinetics.org/api/v1

Main endpoints
--------------

- GET /health/, check service health.
- GET /methods/, list supported methods.
- GET /quota/, check remaining quota.
- POST /validate/, validate input data.
- POST /submit/, submit a prediction job.
- GET /status/<jobId>/, poll job status.
- GET /result/<jobId>/, download results.

Quota
-----

The default quota is 20,000 predictions per day per API key.

Good API practice
-----------------

- Validate input before submission.
- Poll status at a modest interval.
- Download results after completion.
- Store job IDs for audit.
- Cite the underlying model publications in your work.
