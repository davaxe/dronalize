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
| Filtering | Exclude unimportant actors; keep scenes with at least 1 agent; require last observation frame (19) |
| Maps | Relevant area (padding 1.15) |

### Filtering details

| Scope | Rule | Effect |
| ----- | ---- | ------ |
| Cleanup | Exclude categories | Remove unimportant actors. |
| Scene | Minimum agents | Keep scenes with at least 1 retained agent. |
| Agent | Require frame 19 | Keep agents present at the last observation frame. |

## Split support

Use the command below for the most up-to-date split support information for this dataset, including native splits, supported custom modes, and any recommended strategy.

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
