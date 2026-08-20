PYTHON := .venv/bin/python
CLI := .venv/bin/riskaware-eda

.PHONY: help bootstrap env test check demo abc epfl organize

help:
	@printf '%s\n' \
	  'make bootstrap  Create/update the canonical Python environment' \
	  'make env        Inspect compilers, Python, ABC, data, and disk usage' \
	  'make test       Run the unit test suite' \
	  'make check      Run environment checks and tests' \
	  'make demo       Run the synthetic end-to-end pipeline' \
	  'make abc        Build/update Berkeley ABC' \
	  'make epfl       Fetch the 20 small EPFL AIGER circuits' \
	  'make organize   Archive obsolete envs and organize generated files'

bootstrap:
	bash scripts/bootstrap_python.sh

env:
	bash scripts/check_environment.sh

test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m pytest -p no:cacheprovider

check: env test

demo:
	$(CLI) demo --output artifacts/demo

abc:
	bash scripts/setup_abc.sh

epfl:
	bash scripts/fetch_epfl.sh

organize:
	bash scripts/organize_workspace.sh
