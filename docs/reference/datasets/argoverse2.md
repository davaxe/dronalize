# Argoverse 2

<div class="section-intro" markdown="1">
Argoverse 2 expands the original Argoverse benchmark into a broader forecasting and perception dataset family. For motion prediction, it offers larger scale, broader city coverage, and detailed map context across diverse urban driving scenes.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Mixed urban</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Mixed</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Vehicle</strong></div>
  <div class="summary-item"><span>Map context</span><strong>HD</strong></div>
  <div class="summary-item"><span># Samples</span><strong>Processed samples planned</strong></div>
</div>

## Dataset facts

| Field               | Value                                                           | Notes                                                                  |
| ------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Release year        | 2023                                                            | Based on the cited dataset paper and release.                          |
| Domain              | Autonomous-driving forecasting                                  | Designed for large-scale motion prediction and related tasks.          |
| Capture platform    | Self-driving vehicle fleet                                      | Collected from real urban driving.                                     |
| Primary agent types | Vehicles, pedestrians, cyclists, other traffic participants     | Broader participant coverage than many earlier forecasting benchmarks. |
| Map context         | HD maps                                                         | Rich local road geometry is part of the dataset identity.              |
| Geographic coverage | Austin, Detroit, Miami, Pittsburgh, Palo Alto, Washington, D.C. | Built to span multiple North American cities.                          |
| Data format         | Scenario directories                                            | Organized by train, validation, and test scenario folders.             |

## Default processing profile

These are the default Dronalize settings used when processing this dataset.

| Setting | Default |
| ------- | ------- |
| Source sequence | 50 obs / 60 pred @ 10 Hz |
| Effective sequence | 50 obs / 60 pred @ 10 Hz |
| Resampling | None |
| Windowing | None |
| Filtering | Exclude static, unknown, and unimportant actors; require last observation frame (49) |
| Maps | Full map |

### Filtering details

| Scope | Rule | Effect |
| ----- | ---- | ------ |
| Cleanup | Exclude categories | Remove static, unknown, and unimportant actors. |
| Agent | Require frame 49 | Keep agents present at the last observation frame. |

## Split support

Use the command below for the most up-to-date split support information for this dataset, including native splits, supported custom split strategies, and any recommended strategy.

```bash
dronalize split-support argoverse2
```

## References

- Dataset paper: [Argoverse 2: Next Generation Datasets for Self-driving Perception and Forecasting](https://arxiv.org/abs/2301.00493)

## Expected structure

```text
av2/
├── train/
│   ├── ...
│   └── ffffe3df-8d26-42c3-9e7a-59de044736a0/
├── val/
│   ├── ...
│   └── fffc6ef5-8fb4-4f20-afea-b9cb63c99182/
└── test/
    ├── ...
    └── fffc1965-9f9e-4822-ade7-750d87c4b7b9/
```
