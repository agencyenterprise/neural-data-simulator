SHELL=/bin/bash -o pipefail

.PHONY:help lint lint-check test test-converage clean apidoc spellcheck htmldoc run-closed-loop

help:
	@echo "Available commands are: \n*lint, lint-check, test"

lint:
	poetry run black .
	poetry run isort .
	poetry run flake8 .
	poetry run pyright

lint-check:
	poetry run black --check .
	poetry run isort --check .
	poetry run flake8 .
	poetry run pyright --warnings

test:
	poetry run pytest . --color=yes --ignore=experiments --ignore=BKP -m "not jitter" --cov=src/nds --cov-report=term-missing:skip-covered --junitxml=pytest.xml --cov-report=xml 2>&1

test-coverage:
	poetry run pytest . --color=yes --ignore=experiments --ignore=BKP -m "not jitter" --cov=src/nds --cov-report=term-missing:skip-covered --junitxml=pytest.xml --cov-report=xml 2>&1 | tee pytest-coverage.txt

clean:
	-rm docs/source/*.rst
	-rm -rf docs/source/nds
	-rm -rf docs/source/tools
	-rm -rf docs/html
	-rm -rf docs/source/auto_examples

apidoc:
	poetry run sphinx-apidoc -M -H "Neural Data Simulator" -t docs/templates -o docs/source/nds src/nds src/nds/settings.py src/nds/util src/plugins
	poetry run sphinx-apidoc -M -H "Example Implementations and Utilities" -t docs/templates -o docs/source/tools src/ src/nds src/plugins

spellcheck: apidoc
	poetry run sphinx-build -W -b spelling docs/source docs/build

htmldoc: apidoc
	poetry run make -C docs html

run-closed-loop:
	poetry run encoder & poetry run ephys_generator & poetry run decoder & poetry run center_out_reach; pkill -f decoder; pkill -f ephys_generator; pkill -f encoder
