# Argoverse 1

<div class="section-intro" markdown="1">
Argoverse 1 is an early large-scale autonomous-driving forecasting benchmark with rich HD maps. It combines tracked actors and city-scale map context, and it remains a common reference point for map-aware motion prediction.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Mixed urban</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Mixed</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Vehicle</strong></div>
  <div class="summary-item"><span>Map context</span><strong>Limited HD</strong></div>
  <div class="summary-item"><span># Samples</span><strong>Processed samples planned</strong></div>
</div>

## Dataset facts

| Field               | Value                               | Notes                                                             |
| ------------------- | ----------------------------------- | ----------------------------------------------------------------- |
| Release year        | 2019                                | Based on the cited dataset paper and release.                     |
| Domain              | Autonomous-driving forecasting      | Structured around city driving with map-aware prediction tasks.   |
| Capture platform    | Self-driving vehicle fleet          | Collected from on-road autonomous-driving data.                   |
| Primary agent types | Cars, pedestrians, cyclists         | Forecasting is centered on urban traffic participants.            |
| Map context         | Limited HD vector maps              | One of the benchmark's defining features.                         |
| Geographic coverage | Pittsburgh and Miami, United States | Covers two cities with distinct road layouts.                     |
| Data format         | CSV scenes plus map files           | Distributed as split-specific forecasting folders and map assets. |

## Default processing profile

These are the default Dronalize settings used when processing this dataset.

| Setting | Default |
| ------- | ------- |
| Source sequence | 20 obs / 30 pred @ 10 Hz |
| Effective sequence | 20 obs / 30 pred @ 10 Hz |
| Resampling | None |
| Windowing | None |
| Filtering | Require last observation frame (19) |
| Maps | Relevant area (padding 1.15) |

## Split support

Use the command below for the most up-to-date split support information for this dataset, including native splits, supported custom split strategies, and any recommended strategy.

```bash
dronalize split-support argoverse1
```

## References

- Dataset paper: [Argoverse: 3D Tracking and Forecasting with Rich Maps](https://arxiv.org/abs/1911.02620)

## Expected structure

```text
argoverse/
├── forecasting_train_v1.1/
│   └── train/
│       └── data/
│           ├── 1.csv
│           └── ...
├── forecasting_val_v1.1/
├── forecasting_test_v1.1/
└── hd_map/
    └── map_files/
        ├── pruned_argoverse_MIA_10316_vector_map.xml
        └── ...
```
