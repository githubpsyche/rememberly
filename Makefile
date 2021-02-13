.ONESHELL:
SHELL := /bin/bash
SRC = $(wildcard ./*.ipynb)

all: rememberly clean docs

rememberly: $(SRC)
	jupytext --sync *.py 
	jupytext --sync */*.py
	nbdev_build_lib

sync:
	nbdev_update_lib

docs_serve: docs
	cd docs && bundle exec jekyll serve

docs: $(SRC)
	nbdev_build_docs

test:
	nbdev_test_nbs

release: pypi conda_release
	nbdev_bump_version

conda_release:
	fastrelease_conda_package

pypi: dist
	twine upload --repository pypi dist/*

dist: clean
	python setup.py sdist bdist_wheel

clean: $(SRC)
	nbdev_clean_nbs
	nbdev_trust_nbs