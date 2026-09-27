.PHONY: help test lint typecheck environment smoke assets-list dataset-audit docker-build docker-test docker-config provision

help:
	@echo "Targets:"
	@echo "  test            run the full pytest suite"
	@echo "  lint            run ruff check (install the dev extras first)"
	@echo "  typecheck       run mypy over src and tests (install the dev extras first)"
	@echo "  environment     capture the environment report"
	@echo "  smoke           run M0 smoke checks"
	@echo "  assets-list     list registered datasets and models"
	@echo "  dataset-audit   audit the registered MLL23 dataset (needs data/raw/MLL23)"
	@echo "  docker-build    build app and test images (INSTALL_ML=1 for ML extras)"
	@echo "  docker-test     run the full pytest suite in a container"
	@echo "  docker-config   validate the compose file"
	@echo "  provision       provision this machine (MODE=minimal|full, default full)"

test:
	python3 -m pytest

lint:
	python3 -m ruff check .

typecheck:
	PYTHONPATH=src python3 -m mypy src tests

environment:
	PYTHONPATH=src python3 -m bloodfilm.cli environment --output outputs/reports/environment.json

smoke:
	PYTHONPATH=src python3 -m bloodfilm.cli smoke --config configs/base.yaml

assets-list:
	PYTHONPATH=src python3 -m bloodfilm.cli assets list --output outputs/reports/assets.json

dataset-audit:
	PYTHONPATH=src python3 -m bloodfilm.cli dataset audit mll23 --report-output outputs/reports/mll23_audit.json

provision:
	bash scripts/provision_machine.sh "$(CURDIR)"

docker-build:
	docker compose build

docker-test:
	docker compose run --rm test

docker-config:
	docker compose config --quiet
