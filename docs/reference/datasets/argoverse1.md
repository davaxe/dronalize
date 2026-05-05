# Argoverse 1

<div class="section-intro" markdown="1">
Argoverse 1 is an early large-scale autonomous-driving forecasting benchmark with rich HD maps. It combines tracked actors and city-scale map context, and it remains a common reference point for map-aware motion prediction.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Mixed urban</strong></div>
  <div class="summary-item"><span>Release year</span><strong>2019</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Mixed</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Vehicle</strong></div>
  <div class="summary-item"><span>Map context</span><strong>Limited HD</strong></div>
  <div class="summary-item"><span># Samples</span><strong>Processed samples planned</strong></div>
</div>

## Default processing profile

These are the default Dronalize settings used when processing this dataset.

| Setting | Default |
| ------- | ------- |
| Source sequence | 20 obs / 30 pred @ 10 Hz |
| Effective sequence | 20 obs / 30 pred @ 10 Hz |
| Resampling | None |
| Windowing | None |
| Screening | Prune agents with fewer than 2 samples |
| Maps | Relevant area (padding 1.15) |

## Dataset compatibility

Dronalize targets the release or raw layout below. If you have an older or newer download, expect breakage when split names, file names, schemas, or map assets differ.

| Field | Value |
| ----- | ----- |
| Expected release/layout | Argoverse Forecasting v1.1 |
| Loader expectation | The loader expects `forecasting_train_v1.1`, `forecasting_val_v1.1`, and `forecasting_test_v1.1`. |

## Normalization

### Agent categories

| Dataset type | Dronalize type |
| ------------ | -------------- |
| `OBJECT_TYPE = AV` | `CAR` |
| `OBJECT_TYPE = OTHERS` | `UNKNOWN` |

### Map types

| Dataset type | Dronalize type |
| ------------ | -------------- |
| Non-intersection lane segment with left and right neighbors | `LINE_THIN` |
| Non-intersection lane segment without left or right neighbors | `LINE_THIN` |
| Non-intersection outer border without an adjacent neighbor | `CURB` |
| Intersection border | `VIRTUAL` |

## Split support

Use the command below for the most up-to-date split support information for this dataset, including native splits, supported custom split strategies, and any recommended strategy.

```bash
dronalize split-support argoverse1
```

## References

- Dataset paper: [Argoverse: 3D Tracking and Forecasting with Rich Maps](https://arxiv.org/abs/1911.02620)

## Expected structure

```text
argoverse1/
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
