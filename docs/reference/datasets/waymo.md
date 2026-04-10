# Waymo

<div class="section-intro" markdown="1">
The Waymo Open Motion Dataset is a large-scale benchmark for interactive motion forecasting. It is designed around multi-agent behavior in realistic driving scenes, with map context included for local road geometry.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Forecasting</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Mixed</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Vehicle</strong></div>
  <div class="summary-item"><span>Map context</span><strong>HD</strong></div>
  <div class="summary-item"><span># Samples</span><strong>Processed samples planned</strong></div>
</div>

!!! info "Extra dependencies"
    This dataset requires the `waymo` extra to be installed. You can do this with pip (or your package manager of choice):

    ```bash
    pip install dronalize[waymo]
    ``` 

## Dataset facts

| Field               | Value                           | Notes                                                         |
| ------------------- | ------------------------------- | ------------------------------------------------------------- |
| Release year        | 2021                            | Based on the cited dataset paper and release.                 |
| Domain              | Mixed urban autonomous driving  | Designed for planning-relevant multi-agent behavior.          |
| Capture platform    | Self-driving vehicle fleet      | Collected from real autonomous-driving operation.             |
| Primary agent types | Vehicles, pedestrians, cyclists | Covers major urban traffic participants.                      |
| Map context         | Local map geometry              | Each scene includes map information for the surrounding area. |
| Geographic coverage | Waymo Open Dataset cities       | Used as a large-scale self-driving forecasting benchmark.     |
| Data format         | TFRecord scenario files         | Organized into training, validation, and testing releases.    |

## Default processing profile

These are the default Dronalize settings used when processing this dataset.

| Setting | Default |
| ------- | ------- |
| Source sequence | 10 obs / 80 pred @ 10 Hz |
| Effective sequence | 10 obs / 80 pred @ 10 Hz |
| Resampling | None |
| Windowing | None |
| Filtering | Require last observation frame (9) |
| Maps | Full map |

## Split support

Use the command below for the most up-to-date split support information for this dataset, including native splits, supported custom split strategies, and any recommended strategy.

```bash
dronalize split-support waymo
```

## References

- Dataset paper: [Large Scale Interactive Motion Forecasting for Autonomous Driving: The Waymo Open Motion Dataset](https://arxiv.org/abs/2104.10133)

## Expected structure

```text
waymo/
├── training/
│   ├── training.tfrecord-00000-of-01000
│   └── ...
├── validation/
└── testing/
```
