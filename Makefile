PYTHON ?= python

.PHONY: lint format-check format test ci reproduce release-tag

lint:
	$(PYTHON) -m ruff check .

format-check:
	$(PYTHON) -m black --check \
		run_experiment.py \
		trust_region_solver.py \
		trust_region_subproblem.py \
		mip_subproblem.py \
		rt0_dual_solver.py \
		tv_coarse_handler.py \
		synthetic_problem.py \
		utility.py \
		tests

format:
	$(PYTHON) -m black \
		run_experiment.py \
		trust_region_solver.py \
		trust_region_subproblem.py \
		mip_subproblem.py \
		rt0_dual_solver.py \
		tv_coarse_handler.py \
		synthetic_problem.py \
		utility.py \
		tests

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

ci:
	$(PYTHON) -m py_compile *.py tests/*.py
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

reproduce:
	./reproduce.sh

release-tag:
	git tag -a v0.2.0 -m "Release v0.2.0"
	git push origin v0.2.0
