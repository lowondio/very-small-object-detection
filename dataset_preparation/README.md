# dataset_preparation

This package contains the compact pipeline used to build SFO-2class, a merged small-flying-object dataset built from DUT-Anti-UAV and SOD4SB.

## Structure

```text
dataset_preparation/
  __init__.py
  __main__.py
  core.py        geometry, source readers, selection, dataset build, verification
  README.md      package docs
  download_sources.sh
  package.sh
```

## Quick start

```bash
bash dataset_preparation/download_sources.sh data
python -m dataset_preparation analyze --raw data
python -m dataset_preparation build --raw data --out dataset/sfo-2class
python -m dataset_preparation verify --out dataset/sfo-2class --raw data
```

## Design notes

- The geometry policy is in one place: `core.py`.
- Source readers for DUT and SOD4SB stay source-agnostic after conversion to `SourceImage`.
- Dataset proofing is done in the same package, not in a separate hidden dependency chain.
- The package exposes the same core functions as a public API via `dataset_preparation.__init__`.
