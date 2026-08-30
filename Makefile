PYTHON := .venv/bin/python
CLI := .venv/bin/riskaware-eda

.PHONY: help bootstrap env test check demo pilot experiment-plan analyze-full ablation-pilot ablation-full live-pilot abc epfl organize

help:
	@printf '%s\n' \
	  'make bootstrap  Create/update the canonical Python environment' \
	  'make env        Inspect compilers, Python, ABC, data, and disk usage' \
	  'make test       Run the unit test suite' \
	  'make check      Run environment checks and tests' \
	  'make demo       Run the synthetic end-to-end pipeline' \
	  'make pilot      Run or resume the small real-ABC pilot experiment' \
	  'make analyze-full  Audit formal results and build SVG/Markdown report' \
	  'make ablation-pilot Run/resume the six-policy offline ablation pilot' \
	  'make ablation-full  Run/resume the full six-policy ablation sweep' \
	  'make live-pilot  Run/resume the small live-ABC validation pilot' \
	  'make experiment-plan  Validate and print the full experiment plan' \
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

pilot:
	$(CLI) experiment --config configs/pilot_experiment.json --resume

experiment-plan:
	$(CLI) experiment --config configs/experiment.json --dry-run

analyze-full:
	$(PYTHON) scripts/analyze_results.py \
	  --simulation-root artifacts/experiments/epfl_full/simulations \
	  --output-dir artifacts/analysis/epfl_full \
	  --config configs/experiment.json

ablation-pilot:
	$(PYTHON) scripts/run_ablations.py --settings configs/ablation_pilot.json --resume

ablation-full:
	$(PYTHON) scripts/run_ablations.py --settings configs/ablation_full.json --resume

live-pilot:
	$(PYTHON) scripts/validate_live.py --settings configs/live_validation_pilot.json --resume

abc:
	bash scripts/setup_abc.sh

epfl:
	bash scripts/fetch_epfl.sh

organize:
	bash scripts/organize_workspace.sh
