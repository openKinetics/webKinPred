#!/bin/sh
set -eu
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<'SQL'
CREATE DATABASE webkinpred_seqmap;
CREATE DATABASE webkinpred_predictions;
SQL
