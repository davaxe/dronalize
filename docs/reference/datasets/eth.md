# ETH

<div class="section-intro" markdown="1">
ETH is one of the classic pedestrian trajectory benchmarks used in socially aware forecasting. It is usually discussed together with the UCY scenes and is valued for crowd interaction studies rather than map-rich vehicle forecasting.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Pedestrian</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Pedestrians</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Camera</strong></div>
  <div class="summary-item"><span>Map context</span><strong>None</strong></div>
  <div class="summary-item"><span># Samples</span><strong>Processed samples planned</strong></div>
</div>

## Dataset facts

| Field               | Value                               | Notes                                                       |
| ------------------- | ----------------------------------- | ----------------------------------------------------------- |
| Release year        | 2009                                | Based on the cited ETH benchmark paper.                     |
| Domain              | Pedestrian                          | Commonly used for crowd and social-interaction prediction.  |
| Capture platform    | Overhead pedestrian scene recording | Focused on public-space walking behavior.                   |
| Primary agent types | Pedestrians                         | Human motion is the sole prediction target.                 |
| Map context         | Limited                             | The benchmark is usually used without rich map structure.   |
| Benchmark family    | ETH/UCY                             | Closely related to `hotel`, `univ`, `zara1`, and `zara2`.   |
| Data format         | Text trajectory files               | Commonly arranged into train, validation, and test folders. |

## Default processing profile

These are the default Dronalize settings used when processing this dataset.

| Setting | Default |
| ------- | ------- |
| Source sequence | 8 obs / 12 pred @ 2.5 Hz |
| Effective sequence | 29 obs / 48 pred @ 10 Hz |
| Resampling | Linear 4:1 |
| Windowing | 20-frame window, step 1 |
| Filtering | Keep agents with at least 2 samples |
| Maps | Disabled |

## Version

Dronalize does not currently rely on a dataset-specific version marker for ETH. The raw layout used by the loader does not expose a stable release version.

## Normalization

### Agent categories

| Dataset type | Dronalize type | Notes |
| ------------ | -------------- | ----- |
| Any tracked actor | `PEDESTRIAN` | The ETH/UCY loaders normalize every trajectory to the pedestrian category. |

### Map types

| Dataset type | Dronalize type | Notes |
| ------------ | -------------- | ----- |
| Not applicable | Not applicable | Dronalize does not resolve maps for this dataset. |

## Split support

Use the command below for the most up-to-date split support information for this dataset, including native splits, supported custom split strategies, and any recommended strategy.

```bash
dronalize split-support eth
```

## References

- Dataset paper: [You'll never walk alone: Modeling social behavior for multi-target tracking](https://ieeexplore.ieee.org/document/5459260)

## Expected structure

```text
eth/
├── train/
├── val/
└── test/
```

## Notes

- The ETH and UCY scenes are often used as a shared benchmark family, even when individual scenes are reported separately in the literature.
