.PHONY: install install-dev test lint smoke run clean

install:
	pip install -e .

install-dev:
	pip install -e ".[data,dev]"

test:
	pytest tests/ -q

lint:
	ruff check src tests

# ~2 minutes on a laptop CPU: proves the pipeline end to end on synthetic data.
smoke:
	python -m umi.cli run-all --dataset synthetic --n-train-synthetic 800 \
		--epochs 3 --n-members 2 --n-samples 10 --out runs/smoke --no-maps

# The headline experiment (chest X-ray pneumonia).
run:
	python -m umi.cli run-all --dataset pneumoniamnist --epochs 25 \
		--n-members 5 --n-samples 30 --out runs/pneumonia

clean:
	rm -rf runs/ .pytest_cache .ruff_cache **/__pycache__
