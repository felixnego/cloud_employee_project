-- ---------------------------------------------------------------------------
-- The audience schema: SYNTHETIC panel data.
--
-- A separate schema, not a naming convention, because the worst failure mode
-- for this project is a synthetic number being read as a real one. `main` is
-- measured from real video files; `audience` is generated. Every row also
-- carries `data_source = 'synthetic_v1'`.
--
-- These seven tables are the contract (see audience/contract.py). A real
-- ingest would write the same tables and nothing downstream would change.
-- ---------------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS audience;

CREATE OR REPLACE TABLE audience.viewers AS
    SELECT * FROM read_parquet('${AUDIENCE}/viewers.parquet');

CREATE OR REPLACE TABLE audience.sessions AS
    SELECT * FROM read_parquet('${AUDIENCE}/sessions.parquet');

CREATE OR REPLACE TABLE audience.biometric_seconds AS
    SELECT * FROM read_parquet('${AUDIENCE}/biometric_seconds.parquet');

CREATE OR REPLACE TABLE audience.survey_questions AS
    SELECT * FROM read_parquet('${AUDIENCE}/survey_questions.parquet');

CREATE OR REPLACE TABLE audience.survey_responses AS
    SELECT * FROM read_parquet('${AUDIENCE}/survey_responses.parquet');

CREATE OR REPLACE TABLE audience.iat_trials AS
    SELECT * FROM read_parquet('${AUDIENCE}/iat_trials.parquet');

CREATE OR REPLACE TABLE audience.iat_scores AS
    SELECT * FROM read_parquet('${AUDIENCE}/iat_scores.parquet');

-- The coefficients that produced the panel. A real panel has no such table;
-- here it exists so analysis can be checked against ground truth.
CREATE OR REPLACE TABLE audience.generator_params AS
    SELECT * FROM read_parquet('${AUDIENCE}/generator_params.parquet');

CREATE OR REPLACE TABLE audience.generator_manifest AS
    SELECT * FROM read_parquet('${AUDIENCE}/generator_manifest.parquet');
