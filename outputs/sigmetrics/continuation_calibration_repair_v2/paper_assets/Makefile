PYTHON ?= python3
ROOT := $(abspath ..)
export MPLCONFIGDIR ?= /tmp/checkrcq-paper-mpl

.PHONY: all figures tables validate clean-generated

all:
	cd $(ROOT) && $(PYTHON) paper_assets/scripts/generate_all.py

figures:
	cd $(ROOT) && $(PYTHON) paper_assets/scripts/generate_all.py --figures

tables:
	cd $(ROOT) && $(PYTHON) paper_assets/scripts/generate_all.py --tables

validate:
	cd $(ROOT) && $(PYTHON) paper_assets/scripts/validate_assets.py
