# Lyft Level 5

<div class="section-intro" markdown="1">
The Lyft Level 5 motion prediction dataset is a large-scale self-driving benchmark built around long-duration collection, semantic maps, and actor forecasting. It is often cited as one of the major early large-motion datasets for autonomous-driving prediction.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Mixed urban</strong></div>
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

## Dataset facts

| Field               | Value                                | Notes                                                        |
| ------------------- | ------------------------------------ | ------------------------------------------------------------ |
| Release year        | 2021                                 | Based on the cited dataset paper and release.                |
| Domain              | Mixed urban autonomous driving       | Built for large-scale actor forecasting.                     |
| Capture platform    | Self-driving vehicle fleet           | Collected over an extended period on a fixed operating area. |
| Primary agent types | Cars, pedestrians, cyclists          | Covers major traffic participants in urban driving.          |
| Map context         | HD semantic maps                     | Includes both semantic map data and aerial context.          |
| Geographic coverage | Palo Alto, California                | Focused on one self-driving operating area.                  |
| Data format         | Zarr scenes plus semantic map assets | Released as benchmark scene archives and map files.          |

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

## Version

Dronalize does not currently infer a stable upstream release version for the Lyft Level 5 dataset from the raw layout. The loader relies on the `train/train.zarr` and `validate/validate.zarr` structure rather than on a versioned dataset root.

## Normalization

### Agent categories

| Dataset type | Dronalize type | Notes |
| ------------ | -------------- | ----- |
| `label_probabilities argmax = 0` | `UNIMPORTANT` | Lookup entry from `_CATEGORY_LOOKUP`. |
| `label_probabilities argmax = 1` | `UNKNOWN` | Lookup entry from `_CATEGORY_LOOKUP`. |
| `label_probabilities argmax = 2` | `UNIMPORTANT` | Lookup entry from `_CATEGORY_LOOKUP`. |
| `label_probabilities argmax = 3` | `CAR` | Lookup entry from `_CATEGORY_LOOKUP`. |
| `label_probabilities argmax = 4` | `VAN` | Lookup entry from `_CATEGORY_LOOKUP`. |
| `label_probabilities argmax = 5` | `TRAM` | Lookup entry from `_CATEGORY_LOOKUP`. |
| `label_probabilities argmax = 6` | `BUS` | Lookup entry from `_CATEGORY_LOOKUP`. |
| `label_probabilities argmax = 7` | `TRUCK` | Lookup entry from `_CATEGORY_LOOKUP`. |
| `label_probabilities argmax = 8` | `EMERGENCY_VEHICLE` | Lookup entry from `_CATEGORY_LOOKUP`. |
| `label_probabilities argmax = 9` | `UNKNOWN` | Lookup entry from `_CATEGORY_LOOKUP`. |
| `label_probabilities argmax = 10` | `BICYCLE` | Lookup entry from `_CATEGORY_LOOKUP`. |
| `label_probabilities argmax = 11` | `MOTORCYCLE` | Lookup entry from `_CATEGORY_LOOKUP`. |
| `label_probabilities argmax = 12` | `BICYCLE` | Lookup entry from `_CATEGORY_LOOKUP`. |
| `label_probabilities argmax = 13` | `MOTORCYCLE` | Lookup entry from `_CATEGORY_LOOKUP`. |
| `label_probabilities argmax = 14` | `PEDESTRIAN` | Lookup entry from `_CATEGORY_LOOKUP`. |
| `label_probabilities argmax = 15` | `ANIMAL` | Lookup entry from `_CATEGORY_LOOKUP`. |
| `label_probabilities argmax = 16` | `UNIMPORTANT` | Lookup entry from `_CATEGORY_LOOKUP`. |

### Map types

| Dataset type | Dronalize type | Notes |
| ------------ | -------------- | ----- |
| `UNKNOWN` | `NONE` | The parser leaves unknown boundary labels unmapped. |
| `NONE` | `VIRTUAL` | Boundaries explicitly marked as none are treated as virtual. |
| `SINGLE_YELLOW_SOLID` | `LINE_THIN` | Lyft lane-boundary mapping in the parser. |
| `SINGLE_WHITE_SOLID` | `LINE_THIN` | Lyft lane-boundary mapping in the parser. |
| `SINGLE_YELLOW_DASHED` | `LINE_THIN_DASHED` | Lyft lane-boundary mapping in the parser. |
| `SINGLE_WHITE_DASHED` | `LINE_THIN_DASHED` | Lyft lane-boundary mapping in the parser. |
| `DOUBLE_YELLOW_SOLID` | `LINE_THIN_DOUBLE` | Lyft lane-boundary mapping in the parser. |
| `DOUBLE_WHITE_SOLID` | `LINE_THIN_DOUBLE` | Lyft lane-boundary mapping in the parser. |
| `DOUBLE_YELLOW_SOLID_FAR_DASHED_NEAR` | `LINE_THIN_DOUBLE_DASHED` | Lyft lane-boundary mapping in the parser. |
| `DOUBLE_YELLOW_DASHED_FAR_SOLID_NEAR` | `LINE_THIN_DOUBLE_DASHED` | Lyft lane-boundary mapping in the parser. |
| `CURB_RED` | `CURB` | Lyft lane-boundary mapping in the parser. |
| `CURB_YELLOW` | `CURB` | Lyft lane-boundary mapping in the parser. |
| `CURB` | `CURB` | Lyft lane-boundary mapping in the parser. |

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
