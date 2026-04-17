# ExiD

<div class="section-intro" markdown="1">
ExiD is a drone-based highway dataset centered on highly interactive entry and exit scenarios. It extends the naturalistic German drone-dataset family toward merge, diverge, and ramp behaviors that are especially relevant for automated-driving research.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Highway</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Vehicles</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Drone</strong></div>
  <div class="summary-item"><span>Map context</span><strong>HD</strong></div>
  <div class="summary-item"><span># Samples</span><strong>Processed samples planned</strong></div>
</div>

## Dataset facts

| Field               | Value                            | Notes                                                           |
| ------------------- | -------------------------------- | --------------------------------------------------------------- |
| Release year        | 2022                             | Based on the cited dataset paper and release.                   |
| Domain              | Highway entry and exit traffic   | Designed around interactive merge and diverge situations.       |
| Capture platform    | Drone                            | Recorded from an overhead aerial perspective.                   |
| Primary agent types | Cars, trucks, buses, motorcycles | Focused on motorized highway traffic.                           |
| Map context         | Road geometry and lane layout    | Includes map assets for the recording areas.                    |
| Geographic coverage | Germany                          | Spans several highway locations with interaction-heavy traffic. |
| Data format         | CSV trajectories with maps       | Distributed as track files and companion map data.              |

## Default processing profile

These are the default Dronalize settings used when processing this dataset.

| Setting | Default |
| ------- | ------- |
| Source sequence | 50 obs / 125 pred @ 25 Hz |
| Effective sequence | 99 obs / 250 pred @ 50 Hz |
| Resampling | Cubic 2:1 |
| Windowing | 175-frame window, step 25 |
| Filtering | Prune agents with fewer than 2 samples |
| Lane-change sampling | Require 3 lane changes; keep 1 in 3 negatives |
| Maps | Full map |

## Version

Dronalize currently targets exiD `v2.1`, matching the inspected `exiD-dataset-v2.1.zip` distribution name.

## Normalization

### Agent categories

| Dataset type | Dronalize type | Notes |
| ------------ | -------------- | ----- |
| `car` | `CAR` | Shared LevelX category mapping from `class`. |
| `truck` | `TRUCK` | Shared LevelX category mapping from `class`. |
| `bus` | `BUS` | Shared LevelX category mapping from `class`. |
| `trailer` | `TRAILER` | Shared LevelX category mapping from `class`. |
| `motorcycle` | `MOTORCYCLE` | Shared LevelX category mapping from `class`. |
| `bicycle` | `BICYCLE` | Shared LevelX category mapping from `class`. |
| `pedestrian` | `PEDESTRIAN` | Shared LevelX category mapping from `class`. |
| `van` | `VAN` | Shared LevelX category mapping from `class`. |
| `truck_bus` | `TRUCK` | Shared LevelX category mapping that collapses the combined label into `TRUCK`. |
| `animal` | `ANIMAL` | Shared LevelX category mapping from `class`. |

### Map types

| Dataset type | Dronalize type | Notes |
| ------------ | -------------- | ----- |
| `road_border` | `ROAD_BORDER` | Shared Lanelet2/OSM mapping used by the ExiD map builder. |
| `fence` | `ROAD_BORDER` | Shared Lanelet2/OSM mapping used by the ExiD map builder. |
| `wall` | `ROAD_BORDER` | Shared Lanelet2/OSM mapping used by the ExiD map builder. |
| `curbstone` | `CURB` | Shared Lanelet2/OSM mapping used by the ExiD map builder. |
| `stop_line` | `STOP` | Shared Lanelet2/OSM mapping used by the ExiD map builder. |
| `regulatory_element` | `REGULATORY` | Shared Lanelet2/OSM mapping used by the ExiD map builder. |
| `virtual` | `VIRTUAL` | Shared Lanelet2/OSM mapping used by the ExiD map builder. |
| `pedestrian_marking` | `PEDESTRIAN_MARKING` | Shared Lanelet2/OSM mapping used by the ExiD map builder. |
| `bike_marking` | `BIKE_MARKING` | Shared Lanelet2/OSM mapping used by the ExiD map builder. |
| `guard_rail` | `GUARD_RAIL` | Shared Lanelet2/OSM mapping used by the ExiD map builder. |
| `line_thin` with `subtype=dashed` | `LINE_THIN_DASHED` | Shared Lanelet2/OSM mapping used by the ExiD map builder. |
| `line_thin` without `subtype=dashed` | `LINE_THIN` | Shared Lanelet2/OSM mapping used by the ExiD map builder. |
| `line_thick` with `subtype=dashed` | `LINE_THICK_DASHED` | Shared Lanelet2/OSM mapping used by the ExiD map builder. |
| `line_thick` without `subtype=dashed` | `LINE_THICK` | Shared Lanelet2/OSM mapping used by the ExiD map builder. |

## Split support

Use the command below for the most up-to-date split support information for this dataset, including native splits, supported custom split strategies, and any recommended strategy.

```bash
dronalize split-support exid
```

## References

- Dataset paper: [The exiD Dataset: A Real-World Trajectory Dataset of Highly Interactive Highway Scenarios in Germany](https://ieeexplore.ieee.org/document/9827305)

## Expected structure

```text
exid/
├── data/
│   ├── 01_recordingMeta.csv
│   ├── 01_tracks.csv
│   ├── 01_tracksMeta.csv
│   └── ...
└── maps/
    └── ...
```
