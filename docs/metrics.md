# Metrics
This page provides an overview of the evaluation metrics available in the Dronalize toolbox for trajectory prediction.

## Overview
The toolbox includes several evaluation metrics for trajectory prediction, implemented in the [`metrics`](metrics)
module.
The metrics are designed to handle both uni- and multi-modal predictions.
Predictions are expected to be in the form of `(batch_size, num_timesteps, 2)` or
`(batch_size, num_timesteps, num_modes, 2)`, where `num_modes` is the number of modes in the prediction.
> There is also support for mode-first predictions of shape `(batch_size, num_modes, num_timesteps, 2)` that can be used
> by setting the `mode_first` flag to `True`.
> Users can of course change the default behavior by directly modifying the metrics.

## Configuring Metrics
Most metrics are also compsatible with specifying a `min_criterion` (`FDE`, `ADE`, `ML`) that is used to select which of
the modes to evaluate against the ground-truth target (Default: `FDE`).
Setting `min_criterion` to `ML` will evaluate the metrics based on the mode with the highest predicted probability.
Note that `ML` can only be used in conjunction with the optional argument `Prob` of shape `(batch_size, num_modes)`
representing the weights of each mode.

## Available Metrics
The following metrics are implemented:

- [**Min. Average Displacement Error (minADE)**](metrics/min_ade.py)
- [**Min. Final Displacement Error (minFDE)**](metrics/min_fde.py)
- [**Min. Average Path Displacement Error (minAPDE)**](metrics/min_apde.py)
- [**Miss Rate**](metrics/miss_rate.py)
- [**Collision Rate**](metrics/collision_rate.py)
- [**Min. Brier**](metrics/min_brier.py)
- [**Negative Log-Likelihood (NLL)**](metrics/log_likelihood.py)
- [**Expected Displacement Error**](metrics/exp_de.py)

For their mathematical definitions, please refer to the paper.
