# Argoverse 1

Argoverse 1 is an early large-scale autonomous-driving forecasting benchmark with rich HD maps. It combines tracked actors and city-scale map context, and it remains a common reference point for map-aware motion prediction.

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Mixed urban</strong></div>
  <div class="summary-item"><span>Release year</span><strong>2019</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Mixed</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Vehicle</strong></div>
  <div class="summary-item"><span>Map context</span><strong>Limited HD</strong></div>
  <div class="summary-item"><span># Records</span><strong>processed scene records planned</strong></div>
</div>

## Default processing profile

Default processing settings for this dataset.

| Setting | Default |
| ------- | ------- |
| Configured horizon | 50 frames (default split after 20) @ 10.0 Hz |
| Effective horizon | 50 frames (default split after 20) @ 10.0 Hz |
| Source unit | Scenario |
| Source bounds | 20-50 frames (observed) |
| Resampling | None |
| Sliding windows | Disabled |
| Screening | Prune agents with fewer than 2 observations |
| Maps | Trajectory buffer (radius=25) |

## Dataset compatibility

Expected raw data layout for this loader.

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

Use the CLI for current native split and assignment support.

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
