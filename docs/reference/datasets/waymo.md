# Waymo

<div class="section-intro" markdown="1">
The Waymo Open Motion Dataset is a large-scale benchmark for interactive motion forecasting. It is designed around multi-agent behavior in realistic driving scenes, with map context included for local road geometry.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Forecasting</strong></div>
  <div class="summary-item"><span>Release year</span><strong>2021</strong></div>
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

## Default processing profile

These are the default Dronalize settings used when processing this dataset.

| Setting | Default |
| ------- | ------- |
| Configured horizon | 91 frames (default split after 11) @ 10.0 Hz |
| Effective horizon | 91 frames (default split after 11) @ 10.0 Hz |
| Source unit | Scenario |
| Source bounds | 11-91 frames (documented) |
| Resampling | None |
| Sliding windows | Disabled |
| Screening | Prune agents with fewer than 2 samples |
| Maps | Trajectory buffer (radius=25) |

## Dataset compatibility

Dronalize targets the release or raw layout below. If you have an older or newer download, expect breakage when split names, file names, schemas, or map assets differ.

| Field | Value |
| ----- | ----- |
| Expected release/layout | Waymo Motion TFRecord scenario layout |
| Loader expectation | The loader uses the split directory layout and does not infer the exact upstream release revision. |

## Normalization

### Agent categories

| Dataset type | Dronalize type |
| ------------ | -------------- |
| `0` | `UNKNOWN` |
| `1` | `CAR` |
| `2` | `PEDESTRIAN` |
| `3` | `BICYCLE` |
| `4` | `UNKNOWN` |

### Map types

| Dataset type | Dronalize type |
| ------------ | -------------- |
| `RoadLine.TYPE_UNKNOWN` | `VIRTUAL` |
| `RoadLine.TYPE_BROKEN_SINGLE_WHITE` | `LINE_THIN_DASHED` |
| `RoadLine.TYPE_BROKEN_SINGLE_YELLOW` | `LINE_THIN_DASHED` |
| `RoadLine.TYPE_SOLID_SINGLE_WHITE` | `LINE_THIN` |
| `RoadLine.TYPE_SOLID_SINGLE_YELLOW` | `LINE_THIN` |
| `RoadLine.TYPE_SOLID_DOUBLE_WHITE` | `LINE_THIN_DOUBLE` |
| `RoadLine.TYPE_SOLID_DOUBLE_YELLOW` | `LINE_THIN_DOUBLE` |
| `RoadLine.TYPE_PASSING_DOUBLE_YELLOW` | `LINE_THIN_DOUBLE` |
| `RoadLine.TYPE_BROKEN_DOUBLE_YELLOW` | `LINE_THIN_DOUBLE_DASHED` |
| `RoadEdge.TYPE_UNKNOWN` | `VIRTUAL` |
| `RoadEdge.TYPE_ROAD_EDGE_BOUNDARY` | `ROAD_BORDER` |
| `RoadEdge.TYPE_ROAD_EDGE_MEDIAN` | `GUARD_RAIL` |
| Crosswalk polygon | `PEDESTRIAN_MARKING` |
| Speed-bump polygon | `REGULATORY` |
| Lane-center polyline | `VIRTUAL` |
| Driveway polygon | `VIRTUAL` |

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
