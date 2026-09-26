.PHONY: test typecheck environment smoke

test:
	python3 -m pytest

typecheck:
	PYTHONPATH=src python3 -m mypy src tests

environment:
	PYTHONPATH=src python3 -m bloodfilm.cli environment --output outputs/reports/environment.json

smoke:
	PYTHONPATH=src python3 -m bloodfilm.cli smoke --config configs/base.yaml
