# uniD

<div class="section-intro" markdown="1">
uniD is a drone dataset collected in a university-campus environment with strong pedestrian and cyclist activity. It extends the German drone-dataset family into a mixed-traffic setting that is less vehicle-dominated than highway or intersection benchmarks.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Campus</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Mixed</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Drone</strong></div>
  <div class="summary-item"><span>Map context</span><strong>HD</strong></div>
  <div class="summary-item"><span># Samples</span><strong>Processed samples planned</strong></div>
</div>

## Dataset facts

| Field               | Value                                        | Notes                                                                 |
| ------------------- | -------------------------------------------- | --------------------------------------------------------------------- |
| Release year        | 2024                                         | Based on the cited dataset page and release.                          |
| Domain              | Campus mixed traffic                         | Focused on university-road interaction rather than highway-only flow. |
| Capture platform    | Drone                                        | Recorded from an overhead aerial perspective.                         |
| Primary agent types | Cars, trucks or buses, bicycles, pedestrians | Strong emphasis on vulnerable road users.                             |
| Map context         | Campus road layout                           | Includes map material for the recording area.                         |
| Geographic coverage | Germany                                      | Part of the German drone trajectory dataset family.                   |
| Data format         | CSV trajectories with maps                   | Organized into track files and companion map assets.                  |

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

Dronalize currently targets uniD `v1.1`, matching the inspected `uniD-dataset-v1.1.zip` distribution name.

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
| `road_border` | `ROAD_BORDER` | Shared Lanelet2/OSM mapping used by the UniD map builder. |
| `fence` | `ROAD_BORDER` | Shared Lanelet2/OSM mapping used by the UniD map builder. |
| `wall` | `ROAD_BORDER` | Shared Lanelet2/OSM mapping used by the UniD map builder. |
| `curbstone` | `CURB` | Shared Lanelet2/OSM mapping used by the UniD map builder. |
| `stop_line` | `STOP` | Shared Lanelet2/OSM mapping used by the UniD map builder. |
| `regulatory_element` | `REGULATORY` | Shared Lanelet2/OSM mapping used by the UniD map builder. |
| `virtual` | `VIRTUAL` | Shared Lanelet2/OSM mapping used by the UniD map builder. |
| `pedestrian_marking` | `PEDESTRIAN_MARKING` | Shared Lanelet2/OSM mapping used by the UniD map builder. |
| `bike_marking` | `BIKE_MARKING` | Shared Lanelet2/OSM mapping used by the UniD map builder. |
| `guard_rail` | `GUARD_RAIL` | Shared Lanelet2/OSM mapping used by the UniD map builder. |
| `line_thin` with `subtype=dashed` | `LINE_THIN_DASHED` | Shared Lanelet2/OSM mapping used by the UniD map builder. |
| `line_thin` without `subtype=dashed` | `LINE_THIN` | Shared Lanelet2/OSM mapping used by the UniD map builder. |
| `line_thick` with `subtype=dashed` | `LINE_THICK_DASHED` | Shared Lanelet2/OSM mapping used by the UniD map builder. |
| `line_thick` without `subtype=dashed` | `LINE_THICK` | Shared Lanelet2/OSM mapping used by the UniD map builder. |

## Split support

Use the command below for the most up-to-date split support information for this dataset, including native splits, supported custom split strategies, and any recommended strategy.

```bash
dronalize split-support unid
```

## References

- Dataset page: [The uniD Dataset: A university drone dataset](https://levelxdata.com/unid-dataset/)

## Expected structure

```text
unid/
├── data/
│   ├── 01_recordingMeta.csv
│   ├── 01_tracks.csv
│   ├── 01_tracksMeta.csv
│   └── ...
└── maps/
    └── ...
```
