# rounD

<div class="section-intro" markdown="1">
rounD is a naturalistic drone dataset for roundabout traffic in Germany. It is widely used for interaction-heavy road-user forecasting because it captures varied vehicle and vulnerable-road-user behavior in compact circular junctions.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Roundabout</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Mixed</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Drone</strong></div>
  <div class="summary-item"><span>Map context</span><strong>HD</strong></div>
  <div class="summary-item"><span># Samples</span><strong>Processed samples planned</strong></div>
</div>

## Dataset facts

| Field               | Value                              | Notes                                                                 |
| ------------------- | ---------------------------------- | --------------------------------------------------------------------- |
| Release year        | 2020                               | Based on the cited dataset paper and release.                         |
| Domain              | Roundabout traffic                 | Built for dense interaction in circular junctions.                    |
| Capture platform    | Drone                              | Overhead aerial recording reduces occlusion in complex junctions.     |
| Primary agent types | Vehicles and vulnerable road users | Includes cars, trucks, buses, motorcycles, bicycles, and pedestrians. |
| Map context         | Roundabout lane layout             | Includes map material for each recording site.                        |
| Geographic coverage | Germany                            | Recorded at three German roundabouts.                                 |
| Data format         | CSV trajectories with maps         | Organized into trajectory files and lanelet-style map assets.         |

## Default processing profile

These are the default Dronalize settings used when processing this dataset.

| Setting | Default |
| ------- | ------- |
| Source sequence | 50 obs / 125 pred @ 25 Hz |
| Effective sequence | 99 obs / 250 pred @ 50 Hz |
| Resampling | Cubic 2:1 |
| Windowing | 175-frame window, step 25 |
| Filtering | Prune agents with fewer than 2 samples |
| Maps | Full map |

## Version

Dronalize currently targets rounD `v1.1`, matching the inspected `round-dataset-v1.1.zip` distribution name.

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
| `road_border` | `ROAD_BORDER` | Shared Lanelet2/OSM mapping used by the rounD map builder. |
| `fence` | `ROAD_BORDER` | Shared Lanelet2/OSM mapping used by the rounD map builder. |
| `wall` | `ROAD_BORDER` | Shared Lanelet2/OSM mapping used by the rounD map builder. |
| `curbstone` | `CURB` | Shared Lanelet2/OSM mapping used by the rounD map builder. |
| `stop_line` | `STOP` | Shared Lanelet2/OSM mapping used by the rounD map builder. |
| `regulatory_element` | `REGULATORY` | Shared Lanelet2/OSM mapping used by the rounD map builder. |
| `virtual` | `VIRTUAL` | Shared Lanelet2/OSM mapping used by the rounD map builder. |
| `pedestrian_marking` | `PEDESTRIAN_MARKING` | Shared Lanelet2/OSM mapping used by the rounD map builder. |
| `bike_marking` | `BIKE_MARKING` | Shared Lanelet2/OSM mapping used by the rounD map builder. |
| `guard_rail` | `GUARD_RAIL` | Shared Lanelet2/OSM mapping used by the rounD map builder. |
| `line_thin` with `subtype=dashed` | `LINE_THIN_DASHED` | Shared Lanelet2/OSM mapping used by the rounD map builder. |
| `line_thin` without `subtype=dashed` | `LINE_THIN` | Shared Lanelet2/OSM mapping used by the rounD map builder. |
| `line_thick` with `subtype=dashed` | `LINE_THICK_DASHED` | Shared Lanelet2/OSM mapping used by the rounD map builder. |
| `line_thick` without `subtype=dashed` | `LINE_THICK` | Shared Lanelet2/OSM mapping used by the rounD map builder. |

## Split support

Use the command below for the most up-to-date split support information for this dataset, including native splits, supported custom split strategies, and any recommended strategy.

```bash
dronalize split-support round
```

## References

- Dataset paper: [The rounD Dataset: A Drone Dataset of Road User Trajectories at Roundabouts in Germany](https://ieeexplore.ieee.org/document/9294728)

## Expected structure

```text
round/
├── data/
│   ├── 01_recordingMeta.csv
│   ├── 01_tracks.csv
│   ├── 01_tracksMeta.csv
│   └── ...
└── maps/
    └── lanelets/
        └── ...
```
