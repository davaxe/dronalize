# Changelog

All notable changes to Dronalize are documented on this page.

---

## v1.3.0 — August 2025

### Expanded Dataset Support

Added preprocessing pipelines and configuration files for the following datasets:

- *Argoverse 1*
- *Argoverse 2*
- *Waymo Open Motion Dataset (WOMD)*
- *nuScenes*
- *Lyft Level 5 (Woven by Toyota Prediction dataset)*
- *View-of-Delft (VoD)*
- *ApolloScape*
- *I-80* and *US-101* (NGSIM)
- *ETH* and *UCY*
- *openDD*
- *AD4CHE*

### Additional Baseline Model

Added [MTP-GO](https://arxiv.org/abs/2302.00735), a trajectory predictor based on graph-enhanced neural ODEs. This complements the existing prototype model by providing a more advanced approach.

### Updated Dependencies and Containers

Updated `requirements.txt` and container recipes (Docker and Apptainer) for compatibility with latest dataset additions and later versions of PyTorch and PyTorch Geometric. Added a `pyproject.toml` file for better dependency management.

### Improved Documentation

Restructured and extended documentation with modular markdown files, clearer usage instructions, and dataset-specific pages.

---

## v1.2.0 — March 2025

- Added the *INTERACTION* dataset
- Additional attributes in the lane graphs
- Additional visualization tools
- Updated default PyTorch version to 2.5
- Updated requirements and container recipes
- Additional pre-built Docker images (see [Docker Hub](https://hub.docker.com/r/westny/dronalize/tags))

---

## v1.1.1 — December 2024

- Fixed minor bugs in the highway dataset preprocessing
- Minor documentation updates

---

## v1.1.0 — September 2024

- Added 4 new datasets: *exiD*, *uniD*, *SIND*, and *A43*
- Enhanced lane graphs with additional attributes
- Published pre-built Docker image on Docker Hub
- Added PyPI installation support

---

## v1.0.0 — April 2024

- Initial release