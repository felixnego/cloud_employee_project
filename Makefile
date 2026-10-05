# Human interface to the pipeline. Every target is a thin wrapper over
# `docker compose`, so there is nothing to install on the host but Docker.

COMPOSE := docker compose
RUN     := $(COMPOSE) run --rm pipeline

.DEFAULT_GOAL := help
.PHONY: help build list all extract transform docs audience api openapi diagram load-ingested notebook shell clean clean-cache

help:  ## show this help
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

build:  ## build the image (deps + ~680MB of ONNX models, cached after the first run)
	$(COMPOSE) build

list:  ## show the nine extractors and the discovered creatives
	$(RUN) list

all:  ## extract -> transform -> docs  (the whole pipeline)
	$(RUN) all

extract:  ## run the extractors, writing warehouse/raw/*.parquet
	$(RUN) extract

transform:  ## build the DuckDB marts from raw Parquet
	$(RUN) transform

docs:  ## regenerate docs/schema.md from the live warehouse
	$(RUN) docs

audience:  ## generate the synthetic audience panel, load it, and validate it
	$(COMPOSE) run --rm --entrypoint python pipeline -m audience.run all

api:  ## start both APIs: ingest on :8001, processing on :8002 (docs at /docs)
	$(COMPOSE) up ingest-api processing-api

diagram:  ## regenerate docs/architecture.svg
	$(COMPOSE) run --rm --entrypoint python pipeline docs/make_architecture.py

openapi:  ## export both OpenAPI specs to docs/openapi-*.json
	$(COMPOSE) run --rm --entrypoint python pipeline -m api.spec

load-ingested:  ## fold landed biometric batches into the warehouse
	$(COMPOSE) run --rm --entrypoint python pipeline -m api.load

notebook:  ## start JupyterLab on http://localhost:8888  (no token)
	$(COMPOSE) up notebook

shell:  ## open a shell in the pipeline image
	$(COMPOSE) run --rm --entrypoint bash pipeline

clean:  ## delete the warehouse (keeps the decode cache)
	rm -rf warehouse/raw warehouse/marts warehouse/audience warehouse/warehouse.duckdb

clean-cache:  ## delete decoded frames and audio
	rm -rf cache
