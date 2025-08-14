<div align="center">
<img alt="Dronalize logo" src=https://github.com/westny/dronalize/assets/60364134/862a8a60-4cd0-4b21-b0d2-a4ee0e5b4f03 width="800px" style="max-width: 100%;">

______________________________________________________________________

[![Arxiv link](https://img.shields.io/static/v1?label=arXiv&message=Paper&color=8A2BE2&logo=arxiv)](https://arxiv.org/abs/2405.00604)
[![python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch-2.5%20%7C%202.6-blue.svg)](https://pytorch.org/)
[![contributions](https://img.shields.io/badge/Contributions-welcome-297D1E)](#contributing)
[![license](https://img.shields.io/badge/License-Apache%202.0-2F2F2F.svg)](LICENSE)
<br>
[![PyPi Status](https://github.com/westny/dronalize/actions/workflows/pypi.yml/badge.svg)](.github/workflows/pypi.yml)
[![Docker Status](https://github.com/westny/dronalize/actions/workflows/docker-image.yml/badge.svg)](.github/workflows/docker-image.yml)
[![Apptainer Status](https://github.com/westny/dronalize/actions/workflows/apptainer-image.yml/badge.svg)](.github/workflows/apptainer-image.yml)
[![Conda Status](https://github.com/westny/dronalize/actions/workflows/conda.yml/badge.svg)](.github/workflows/conda.yml)

</div>

**Dronalize** is a toolbox designed to streamline the development process for researchers working with trajectory datasets in behavior prediction problems.
Originally developed for drone-captured bird’s-eye-view datasets, it has since evolved to support a wide range of popular benchmarks in motion forecasting.
The toolbox provides utilities for data preprocessing, visualization, and evaluation, along with a model development pipeline tailored for ML-based trajectory prediction.
<br> The toolbox
utilizes [<img alt="Pytorch logo" src=https://github.com/westny/dronalize/assets/60364134/b6d458a5-0130-4f93-96df-71374c2de579 height="12">PyTorch](https://pytorch.org/docs/stable/index.html), [<img alt="PyG logo" src=https://github.com/westny/dronalize/assets/60364134/53554175-0ca1-4020-b8eb-7bbd4ebe0e47 height="12">PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/),
and [<img alt="Lightning logo" src=https://github.com/westny/dronalize/assets/60364134/167a7cbb-8346-44ac-9428-5f963ada54c2 height="16">PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
for its functionality.

### 📝 Changelog

### v1.3.0 (August 2025)

- **Expanded dataset support**  
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

- **Additional baseline model**  
  Added [MTP-GO](https://arxiv.org/abs/2302.00735), a trajectory predictor based on graph-enhanced neural ODEs. This complements the existing prototype model by providing a more advanced approach.

- **Updated dependencies and containers**  
  Updated `requirements.txt` and container recipes (Docker and Apptainer) for compatibility with latest dataset additions and later versions of PyTorch and PyTorch Geometric. Added a `pyproject.toml` file for better dependency management.

- **Improved documentation**  
  Restructured and extended documentation with modular markdown files, clearer usage instructions, and dataset-specific pages.
<details>
  <summary>Click here for full changelog</summary>

### v1.2.0 (March 2025)

- Added the *INTERACTION* dataset
- Additional attributes in the lane graphs
- Additional visualization tools
- Updated default PyTorch version to 2.5
- Updated requirements and container recipes
- Additional pre-built Docker images (see [Docker Hub](https://hub.docker.com/r/westny/dronalize/tags))

### v1.1.1 (December 2024)

- Fixed minor bugs in the highway dataset preprocessing
- Minor documentation updates

### v1.1.0 (September 2024)

- Added 4 new datasets: *exiD*, *uniD*, *SIND*, and *A43*
- Enhanced lane graphs with additional attributes
- Published pre-built Docker image on Docker Hub
- Added PyPi installation support

### v1.0.0 (April 2024)

- Initial release

</details>

<br>

# Documentation Overview
The **Dronalize** toolbox provides a streamlined preprocessing and evaluation pipeline for working with trajectory prediction datasets. While some components are described in the main README, detailed documentation are split into topic-specific pages:

- [Installation Guide](docs/installation.md)
- [Datasets](docs/datasets.md)
- [Preprocessing](docs/preprocessing.md)
- [Modeling](docs/modeling.md)
- [Data Loading](docs/data_loading.md)
- [Training](docs/training.md)
- [Metrics](docs/metrics.md)
- [Related Work](docs/related_work.md)
- [Contributing](docs/contributing.md)
- [Cite](docs/cite.md)

<br>

# Datasets

Initially developed for drone-captured datasets, the toolbox has since been extended to support a broader range of popular trajectory prediction benchmarks. Most are freely available for non-commercial use, although access may require an application or account registration.

A summary of the datasets currently supported by the toolbox is provided below, including links to their respective download pages and brief descriptions of their characteristics.
For detailed usage instructions, folder structure expectations, and specific commands for each supported dataset, please refer to the [**Datasets**](docs/datasets.md) page.



<div align="center">

| Dataset         | Year | Link                                             | Location                                | Acquisition          | Map Info               | Notes            |
|-----------------|------|--------------------------------------------------|-----------------------------------------|----------------------|------------------------|------------------|
| *I-80*          | 2006 | [https://data.transportation.gov/NGSIM](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj/about_data)                 | Interstate 80, USA                      | Static camera        | Limited (lane ids)     | Vehicles         |
| *US-101*        | 2007 | [https://data.transportation.gov/NGSIM](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj/about_data)                | US Highway 101, USA                     | Static camera        | Limited (lane ids)     | Vehicles         |
| *UCY*           | 2007 | https://www.dropbox.com/s/8n02xqv3l9q18r1        | Cyprus                                  | Static camera        | ✗                      | Pedestrians      |
| *ETH*           | 2009 | https://www.dropbox.com/s/8n02xqv3l9q18r1        | Zurich, Switzerland                     | Static camera        | ✗                      | Pedestrians      |
| *highD*         | 2018 | https://levelxdata.com/highd-dataset/            | Highways (Germany)                      | Drone                | Limited (lane lines)   | Vehicles         |
| *ApolloScape*   | 2019 | https://apolloscape.auto/trajectory.html         | Beijing, China                          | Instrumented vehicle | ✗                      | Mixed traffic    |
| *Argoverse*     | 2019 | https://www.argoverse.org/av1.html               | Miami, Pittsburgh, USA                  | Instrumented vehicle | ✓                      | Mixed traffic    |
| *INTERACTION*   | 2019 | https://interaction-dataset.com/                 | USA, China, Germany, Bulgaria           | Drone                | ✓                      | Mixed traffic    |
| *nuScenes*      | 2020 | https://www.nuscenes.org/nuscenes                | Boston (USA), Singapore                 | Instrumented vehicle | ✓                      | Mixed traffic    |
| *inD*           | 2020 | https://levelxdata.com/ind-dataset/              | Intersections (Germany)                 | Drone                | ✓                      | Mixed traffic    |
| *rounD*         | 2020 | https://levelxdata.com/round-dataset/            | Roundabouts (Germany)                   | Drone                | ✓                      | Mixed traffic    |
| *openDD*        | 2020 | https://l3pilot.eu/data/opendd.html              | Roundabouts (Germany)                   | Drone                | ✓                      | Mixed traffic    |
| *Lyft Level 5*  | 2021 | https://woven.toyota/en/prediction-dataset       | Palo Alto, USA                          | Instrumented vehicle | ✓                      | Mixed traffic    |
| *WOMD*          | 2021 | https://waymo.com/open/                          | Six US cities                           | Instrumented vehicle | ✓                      | Mixed traffic    |
| *exiD*          | 2022 | https://levelxdata.com/exid-dataset/             | Highways (Germany)                      | Drone                | ✓                      | Vehicles         |
| *SIND*          | 2022 | https://github.com/SOTIF-AVLab/SinD              | Signalized intersections (China)        | Drone + camera       | ✓                      | Mixed traffic    |
| *Argoverse 2*   | 2023 | https://www.argoverse.org/av2.html               | Six US cities                           | Instrumented vehicle | ✓                      | Mixed traffic    |
| *AD4CHE*        | 2023 | https://github.com/ADSafetyJointLab/AD4CHE       | Highways (China)                        | Drone                | Limited (images)       | Vehicles         |
| *uniD*          | 2024 | https://levelxdata.com/unid-dataset/             | University campus (Germany)             | Drone                | ✓                      | Mixed traffic    |
| *A43*           | 2024 | https://data.isac.rwth-aachen.de/?p=58           | A43 Highway, Germany                    | Drone                | Limited (lane lines)   | Vehicles         |
| *View-of-Delft* | 2024 | https://intelligent-vehicles.org/datasets/view-of-delft/   | Delft, Netherlands                      | Instrumented vehicle | ✓                      | Mixed traffic    |


</div>

> Several datasets in the leveLXData suite were recently updated (April 2024) that include improvements to the maps, as
> well as the addition of some new locations.
> This toolbox is designed to work with the updated datasets, and we recommend using the latest versions for the most
> recent features to avoid having to modify the toolbox.

<br>

# Example Use-Cases

In our accompanying [paper](https://arxiv.org/abs/2405.00604), we outline several use cases for the toolbox and discuss future research directions. One key application is leveraging the toolbox’s general-purpose design for domain adaptation and generalization across different datasets.

The figure below illustrates the performance gains achievable through transfer learning, using the prototype model with default hyperparameters. Specifically, the model is first pre-trained on the INTERACTION dataset and fine-tuned on the inD dataset, and vice versa. As shown, this can lead to significant improvements in performance, even with a very simple model architecture.

<div align="center">
  <img src=https://github.com/user-attachments/assets/56f86808-ac85-47e5-bc50-3a9915cc9d1b alt="transfer_learning.png">
</div>

<br>

# Related work

We have been working with the datasets in several research projects, resulting in multiple published papers focused on
behavior prediction.
If you're interested in learning more about our findings, please refer to the following publications:

#### [Diffusion-Based Environment-Aware Trajectory Prediction](https://arxiv.org/abs/2403.11643)

- **Authors:** Theodor Westny, Björn Olofsson, and Erik Frisk
- **Published In:** ArXiv preprint arXiv:2403.11643

<details>
  <summary>Abstract</summary>
    <p style="font-style: italic;">
    The ability to predict the future trajectories of traffic participants is crucial for the safe and efficient operation of autonomous vehicles.
    In this paper, a diffusion-based generative model for multi-agent trajectory prediction is proposed.
    The model is capable of capturing the complex interactions between traffic participants and the environment, accurately learning the multimodal nature of the data.
    The effectiveness of the approach is assessed on large-scale datasets of real-world traffic scenarios, showing that our model outperforms several well-established methods in terms of prediction accuracy.
    By the incorporation of differential motion constraints on the model output, we illustrate that our model is capable of generating a diverse set of realistic future trajectories.
    Through the use of an interaction-aware guidance signal, we further demonstrate that the model can be adapted to predict the behavior of less cooperative agents, emphasizing its practical applicability under uncertain traffic conditions.
    </p>
</details>

<details>
  <summary>Bibtex</summary>

    @article{westny2024diffusion,
      title={Diffusion-Based Environment-Aware Trajectory Prediction},
      author={Westny, Theodor and Olofsson, Bj{\"o}rn and Frisk, Erik},
      journal={arXiv preprint arXiv:2403.11643},
      year={2024}
    }

</details>

#### [MTP-GO: Graph-Based Probabilistic Multi-Agent Trajectory Prediction with Neural ODEs](https://arxiv.org/abs/2302.00735)

- **Authors:** Theodor Westny, Joel Oskarsson, Björn Olofsson, and Erik Frisk
- **Published In:** 2023 IEEE Transactions on Intelligent Vehicles, Vol. 8, No. 9

<details>
  <summary>Abstract</summary>
    <p style="font-style: italic;">
    Enabling resilient autonomous motion planning requires robust predictions of surrounding road users' future behavior.
    In response to this need and the associated challenges, we introduce our model titled MTP-GO.
    The model encodes the scene using temporal graph neural networks to produce the inputs to an underlying motion model.
    The motion model is implemented using neural ordinary differential equations where the state-transition functions are learned with the rest of the model.
    Multimodal probabilistic predictions are obtained by combining the concept of mixture density networks and Kalman filtering.
    The results illustrate the predictive capabilities of the proposed model across various data sets, outperforming several state-of-the-art methods on a number of metrics.
    </p>
</details>

<details>
  <summary>Bibtex</summary>

    @article{westny2023mtp,
      title="{MTP-GO}: Graph-Based Probabilistic Multi-Agent Trajectory Prediction with Neural {ODEs}",
      author={Westny, Theodor and Oskarsson, Joel and Olofsson, Bj{\"o}rn and Frisk, Erik},
      journal={IEEE Transactions on Intelligent Vehicles},
      year={2023},
      volume={8},
      number={9},
      pages={4223-4236},
      doi={10.1109/TIV.2023.3282308}
    }

</details>

#### [Evaluation of Differentially Constrained Motion Models for Graph-Based Trajectory Prediction](https://arxiv.org/abs/2304.05116)

- **Authors:** Theodor Westny, Joel Oskarsson, Björn Olofsson, and Erik Frisk
- **Published In:** In 2023 IEEE Intelligent Vehicles Symposium (IV)

<details>
  <summary>Abstract</summary>
    <p style="font-style: italic;">
    Given their flexibility and encouraging performance, deep-learning models are becoming standard for motion prediction in autonomous driving.
    However, with great flexibility comes a lack of interpretability and possible violations of physical constraints.
    Accompanying these data-driven methods with differentially-constrained motion models to provide physically feasible trajectories is a promising future direction.
    The foundation for this work is a previously introduced graph-neural-network-based model, MTP-GO.
    The neural network learns to compute the inputs to an underlying motion model to provide physically feasible trajectories.
    This research investigates the performance of various motion models in combination with numerical solvers for the prediction task.
    The study shows that simpler models, such as low-order integrator models, are preferred over more complex, e.g., kinematic models, to achieve accurate predictions.
    Further, the numerical solver can have a substantial impact on performance, advising against commonly used first-order methods like Euler forward.
    Instead, a second-order method like Heun’s can greatly improve predictions.
    </p>
</details>

<details>
  <summary>Bibtex</summary>

    @inproceedings{westny2023eval,
      title={Evaluation of Differentially Constrained Motion Models for Graph-Based Trajectory Prediction},
      author={Westny, Theodor and Oskarsson, Joel and Olofsson, Bj{\"o}rn and Frisk, Erik},
      booktitle={IEEE Intelligent Vehicles Symposium (IV)},
      pages={},
      year={2023},
      doi={10.1109/IV55152.2023.10186615}
    }

</details>

## Contributing

We welcome contributions to the toolbox, and we encourage you to submit pull requests with new features, bug fixes, or
improvements.
Any form of collaboration is appreciated, and we are open to suggestions for new features or changes to the existing
codebase.
Please direct your inquiries to the authors of the paper.

## Cite

If you use the toolbox in your research, please consider citing the paper:

```
@inproceedings{westny2025dronalize,
  title={Toward Unified Practices in Trajectory Prediction Research on Bird's-Eye-View Datasets},
  author={Westny, Theodor and Olofsson, Bj{\"o}rn and Frisk, Erik},
  booktitle={IEEE Intelligent Vehicles Symposium (IV)},
  year={2025}
}
```

Feel free [email us](mailto:theodor.westny@liu.se) if you have any questions or notice any issues with the toolbox.
If you have any suggestions for improvements or new features, we would be happy to hear from you.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
