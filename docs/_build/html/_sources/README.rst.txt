Documentation Notes
===================

This directory contains the Sphinx documentation source for Open Kinetics
Predictor.

Application setup
-----------------

Requirements:

- Python 3.11 or newer.
- Docker and Docker Compose for the normal development stack.
- ``DJANGO_SECRET_KEY`` set in your environment or ``.env`` file.

Generate a local secret key if needed:

.. code-block:: bash

   openssl rand -hex 50

Start the local stack:

.. code-block:: bash

   cp .env.example .env
   docker compose up -d --build

The web interface runs at ``http://localhost:3000`` in the development stack,
and backend API routes are served by Django under ``/api/``.

Build these docs locally
------------------------

.. code-block:: bash

   cd docs
   python -m pip install -r requirements.txt
   make clean html

Open ``docs/_build/html/index.html`` in your browser.

Where to start
--------------

- :doc:`project/overview` introduces the platform.
- :doc:`project/supported_methods` lists prediction targets and engines.
- :doc:`project/contributing` explains how to add a new prediction method.
- :doc:`api_reference` contains generated Python API pages.
