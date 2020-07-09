.PHONY: init test lint pretty precommit_install

BIN = .venv/bin/
CODE = soccer_xg
PYTHON_INTERPRETER = .venv/bin/python

init:
	python3 -m venv .venv
	poetry install

tests/data/spadl-statsbomb-WC-2018.h5:
	$(PYTHON_INTERPRETER) tests/data/download.py

test: tests/data/spadl-statsbomb-WC-2018.h5
	$(BIN)pytest --verbosity=2 --showlocals --strict --log-level=DEBUG --cov=$(CODE) $(args)

lint:
	$(BIN)flake8 --jobs 4 --statistics --show-source $(CODE) tests
	$(BIN)pylint --jobs 4 --rcfile=setup.cfg $(CODE)
	$(BIN)mypy $(CODE) tests
	$(BIN)black --py36 --skip-string-normalization --line-length=79 --check $(CODE) tests
	$(BIN)pytest --dead-fixtures --dup-fixtures

pretty:
	$(BIN)isort --apply --recursive $(CODE) tests
	$(BIN)black --target-version py36 --skip-string-normalization --line-length=79 $(CODE) tests
	$(BIN)unify --in-place --recursive $(CODE) tests

precommit_install:
	echo '#!/bin/sh\nmake test\n' > .git/hooks/pre-commit
	chmod +x .git/hooks/pre-commit

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
