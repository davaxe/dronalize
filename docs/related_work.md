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

```bibtex
    @article{westny2024diffusion,
      title={Diffusion-Based Environment-Aware Trajectory Prediction},
      author={Westny, Theodor and Olofsson, Bj{\"o}rn and Frisk, Erik},
      journal={arXiv preprint arXiv:2403.11643},
      year={2024}
    }
```

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

```bibtex
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
```

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

```bibtex
    @inproceedings{westny2023eval,
      title={Evaluation of Differentially Constrained Motion Models for Graph-Based Trajectory Prediction},
      author={Westny, Theodor and Oskarsson, Joel and Olofsson, Bj{\"o}rn and Frisk, Erik},
      booktitle={IEEE Intelligent Vehicles Symposium (IV)},
      pages={},
      year={2023},
      doi={10.1109/IV55152.2023.10186615}
    }
```

</details>