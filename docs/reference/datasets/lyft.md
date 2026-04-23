# Lyft Level 5

<div class="section-intro" markdown="1">
The Lyft Level 5 motion prediction dataset is a large-scale self-driving benchmark built around long-duration collection, semantic maps, and actor forecasting. It is often cited as one of the major early large-motion datasets for autonomous-driving prediction.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Mixed urban</strong></div>
  <div class="summary-item"><span>Release year</span><strong>2021</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Mixed</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Vehicle</strong></div>
  <div class="summary-item"><span>Map context</span><strong>HD</strong></div>
  <div class="summary-item"><span># Samples</span><strong>Processed samples planned</strong></div>
</div>

!!! info "Extra dependencies"
    This dataset requires the `lyft` extra to be installed. You can do this with pip (or your package manager of choice):

    ```bash
    pip install dronalize[lyft]
    ```

## Default processing profile

These are the default Dronalize settings used when processing this dataset.

| Setting | Default |
| ------- | ------- |
| Source sequence | 20 obs / 50 pred @ 10 Hz |
| Effective sequence | 20 obs / 50 pred @ 10 Hz |
| Resampling | None |
| Windowing | 70-frame window, step 20 |
| Filtering | Prune agents with fewer than 2 samples |
| Maps | Relevant area (padding 1.15) |

## Dataset compatibility

Dronalize targets the release or raw layout below. If you have an older or newer download, expect breakage when split names, file names, schemas, or map assets differ.

| Field | Value |
| ----- | ----- |
| Expected release/layout | Lyft Level 5 prediction Zarr layout |
| Loader expectation | The loader expects `train/train.zarr` and `validate/validate.zarr`, not a versioned dataset root. |

## Normalization

### Agent categories

| Dataset type | Dronalize type |
| ------------ | -------------- |
| `label_probabilities argmax = 0` | `UNIMPORTANT` |
| `label_probabilities argmax = 1` | `UNKNOWN` |
| `label_probabilities argmax = 2` | `UNIMPORTANT` |
| `label_probabilities argmax = 3` | `CAR` |
| `label_probabilities argmax = 4` | `VAN` |
| `label_probabilities argmax = 5` | `TRAM` |
| `label_probabilities argmax = 6` | `BUS` |
| `label_probabilities argmax = 7` | `TRUCK` |
| `label_probabilities argmax = 8` | `EMERGENCY_VEHICLE` |
| `label_probabilities argmax = 9` | `UNKNOWN` |
| `label_probabilities argmax = 10` | `BICYCLE` |
| `label_probabilities argmax = 11` | `MOTORCYCLE` |
| `label_probabilities argmax = 12` | `BICYCLE` |
| `label_probabilities argmax = 13` | `MOTORCYCLE` |
| `label_probabilities argmax = 14` | `PEDESTRIAN` |
| `label_probabilities argmax = 15` | `ANIMAL` |
| `label_probabilities argmax = 16` | `UNIMPORTANT` |

### Map types

| Dataset type | Dronalize type |
| ------------ | -------------- |
| `UNKNOWN` | `NONE` |
| `NONE` | `VIRTUAL` |
| `SINGLE_YELLOW_SOLID` | `LINE_THIN` |
| `SINGLE_WHITE_SOLID` | `LINE_THIN` |
| `SINGLE_YELLOW_DASHED` | `LINE_THIN_DASHED` |
| `SINGLE_WHITE_DASHED` | `LINE_THIN_DASHED` |
| `DOUBLE_YELLOW_SOLID` | `LINE_THIN_DOUBLE` |
| `DOUBLE_WHITE_SOLID` | `LINE_THIN_DOUBLE` |
| `DOUBLE_YELLOW_SOLID_FAR_DASHED_NEAR` | `LINE_THIN_DOUBLE_DASHED` |
| `DOUBLE_YELLOW_DASHED_FAR_SOLID_NEAR` | `LINE_THIN_DOUBLE_DASHED` |
| `CURB_RED` | `CURB` |
| `CURB_YELLOW` | `CURB` |
| `CURB` | `CURB` |

## Split support

Use the command below for the most up-to-date split support information for this dataset, including native splits, supported custom split strategies, and any recommended strategy.

```bash
dronalize split-support lyft
```

## References

- Dataset paper: [One Thousand and One Hours: Self-driving Motion Prediction Dataset](https://proceedings.mlr.press/v155/houston21a.html)

## Expected structure

```text
lyft/
├── semantic_map/
│   ├── semantic_map.pb
│   └── ...
├── train/
│   ├── train.zarr
│   └── ...
└── validate/
    ├── validate.zarr
    └── ...
```
