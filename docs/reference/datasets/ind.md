# inD

<div class="section-intro" markdown="1">
inD is a naturalistic urban-intersection dataset captured from drones above German intersections. It extends the drone-dataset style beyond highways and emphasizes multimodal interaction among vehicles, cyclists, and pedestrians.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Urban</strong></div>
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
| Filtering | Prune agents with fewer than 2 samples |
| Maps | Full map |

## Dataset compatibility

Dronalize targets the release or raw layout below. If you have an older or newer download, expect breakage when split names, file names, schemas, or map assets differ.

| Field | Value |
| ----- | ----- |
| Expected release/layout | inD v1.1 |
| Loader expectation | The loader assumes the inD v1.1 distribution layout. |

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
dronalize split-support ind
```

## References

- Dataset paper: [The inD Dataset: A Drone Dataset of Naturalistic Road User Trajectories at German Intersections](https://arxiv.org/abs/1911.07602)

## Expected structure

```text
ind/
├── data/
│   ├── 01_recordingMeta.csv
│   ├── 01_tracks.csv
│   ├── 01_tracksMeta.csv
│   └── ...
└── maps/
    └── ...
```
