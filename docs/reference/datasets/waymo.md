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
| Source sequence | 11 obs / 80 pred @ 10 Hz |
| Effective sequence | 11 obs / 80 pred @ 10 Hz |
| Resampling | None |
| Windowing | None |
| Filtering | Prune agents with fewer than 2 samples |
| Maps | Relevant area (padding 1.15) |

## Version

Dronalize does not currently infer a stable Motion Dataset release number from the Waymo raw layout. The TFRecord directory structure identifies the split layout, but not the exact upstream release revision.

## Normalization

### Agent categories

| Dataset type | Dronalize type | Notes |
| ------------ | -------------- | ----- |
| `0` | `UNKNOWN` | Waymo object type mapping from the lean scenario proto. |
| `1` | `CAR` | Waymo object type mapping from the lean scenario proto. |
| `2` | `PEDESTRIAN` | Waymo object type mapping from the lean scenario proto. |
| `3` | `BICYCLE` | Waymo object type mapping from the lean scenario proto. |
| `4` | `UNKNOWN` | The current loader does not assign a more specific shared category for this object type. |

### Map types

| Dataset type | Dronalize type | Notes |
| ------------ | -------------- | ----- |
| `RoadLine.TYPE_UNKNOWN` | `VIRTUAL` | Waymo road-line mapping in the map builder. |
| `RoadLine.TYPE_BROKEN_SINGLE_WHITE` | `LINE_THIN_DASHED` | Waymo road-line mapping in the map builder. |
| `RoadLine.TYPE_BROKEN_SINGLE_YELLOW` | `LINE_THIN_DASHED` | Waymo road-line mapping in the map builder. |
| `RoadLine.TYPE_SOLID_SINGLE_WHITE` | `LINE_THIN` | Waymo road-line mapping in the map builder. |
| `RoadLine.TYPE_SOLID_SINGLE_YELLOW` | `LINE_THIN` | Waymo road-line mapping in the map builder. |
| `RoadLine.TYPE_SOLID_DOUBLE_WHITE` | `LINE_THIN_DOUBLE` | Waymo road-line mapping in the map builder. |
| `RoadLine.TYPE_SOLID_DOUBLE_YELLOW` | `LINE_THIN_DOUBLE` | Waymo road-line mapping in the map builder. |
| `RoadLine.TYPE_PASSING_DOUBLE_YELLOW` | `LINE_THIN_DOUBLE` | Waymo road-line mapping in the map builder. |
| `RoadLine.TYPE_BROKEN_DOUBLE_YELLOW` | `LINE_THIN_DOUBLE_DASHED` | Waymo road-line mapping in the map builder. |
| `RoadEdge.TYPE_UNKNOWN` | `VIRTUAL` | Waymo road-edge mapping in the map builder. |
| `RoadEdge.TYPE_ROAD_EDGE_BOUNDARY` | `ROAD_BORDER` | Waymo road-edge mapping in the map builder. |
| `RoadEdge.TYPE_ROAD_EDGE_MEDIAN` | `GUARD_RAIL` | Waymo road-edge mapping in the map builder. |
| Crosswalk polygon | `PEDESTRIAN_MARKING` | The builder emits crosswalk polygons as pedestrian markings. |
| Speed-bump polygon | `REGULATORY` | The builder emits speed bumps as regulatory map elements. |
| Lane-center polyline | `VIRTUAL` | Lane centerlines are represented as virtual edges. |
| Driveway polygon | `VIRTUAL` | Driveways are represented as virtual edges. |

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
