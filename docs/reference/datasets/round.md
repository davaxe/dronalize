# rounD

<div class="section-intro" markdown="1">
rounD is a naturalistic drone dataset for roundabout traffic in Germany. It is widely used for interaction-heavy road-user forecasting because it captures varied vehicle and vulnerable-road-user behavior in compact circular junctions.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Roundabout</strong></div>
  <div class="summary-item"><span>Release year</span><strong>2020</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Mixed</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Drone</strong></div>
  <div class="summary-item"><span>Map context</span><strong>HD</strong></div>
  <div class="summary-item"><span># Samples</span><strong>Processed samples planned</strong></div>
</div>

## Default processing profile

These are the default Dronalize settings used when processing this dataset.

| Setting | Default |
| ------- | ------- |
| Source sequence | 50 obs / 125 pred @ 25 Hz |
| Effective sequence | 99 obs / 250 pred @ 50 Hz |
| Resampling | Cubic 2:1 |
| Windowing | 175-frame window, step 25 |
| Screening | Prune agents with fewer than 2 samples |
| Maps | Full map |

## Dataset compatibility

Dronalize targets the release or raw layout below. If you have an older or newer download, expect breakage when split names, file names, schemas, or map assets differ.

| Field | Value |
| ----- | ----- |
| Expected release/layout | rounD v1.1 |
| Loader expectation | The loader assumes the rounD v1.1 distribution layout. |

## Normalization

### Agent categories

| Dataset type | Dronalize type |
| ------------ | -------------- |
| `car` | `CAR` |
| `truck` | `TRUCK` |
| `bus` | `BUS` |
| `trailer` | `TRAILER` |
| `motorcycle` | `MOTORCYCLE` |
| `bicycle` | `BICYCLE` |
| `pedestrian` | `PEDESTRIAN` |
| `van` | `VAN` |
| `truck_bus` | `TRUCK` |
| `animal` | `ANIMAL` |

### Map types

| Dataset type | Dronalize type |
| ------------ | -------------- |
| `road_border` | `ROAD_BORDER` |
| `fence` | `ROAD_BORDER` |
| `wall` | `ROAD_BORDER` |
| `curbstone` | `CURB` |
| `stop_line` | `STOP` |
| `regulatory_element` | `REGULATORY` |
| `virtual` | `VIRTUAL` |
| `pedestrian_marking` | `PEDESTRIAN_MARKING` |
| `bike_marking` | `BIKE_MARKING` |
| `guard_rail` | `GUARD_RAIL` |
| `line_thin` with `subtype=dashed` | `LINE_THIN_DASHED` |
| `line_thin` without `subtype=dashed` | `LINE_THIN` |
| `line_thick` with `subtype=dashed` | `LINE_THICK_DASHED` |
| `line_thick` without `subtype=dashed` | `LINE_THICK` |

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
