# ExiD

ExiD is a drone-based highway dataset centered on highly interactive entry and exit scenarios. It extends the naturalistic German drone-dataset family toward merge, diverge, and ramp behaviors that are especially relevant for automated-driving research.

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Highway</strong></div>
  <div class="summary-item"><span>Release year</span><strong>2022</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Vehicles</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Drone</strong></div>
  <div class="summary-item"><span>Map context</span><strong>HD</strong></div>
  <div class="summary-item"><span># Records</span><strong>processed scene records planned</strong></div>
</div>

## Default prediction task

The `benchmark` prediction task is selected automatically. In the project config, select it explicitly with `task = "benchmark"` or disable prediction bounds with `task = "none"`.

| Setting | Default |
| ------- | ------- |
| Configured horizon | 175 frames (benchmark origin 50) @ 25.0 Hz |
| Effective horizon | 70 frames (benchmark origin 20) @ 10.0 Hz |
| Source unit | Recording |
| Source bounds | 1332-43534 frames (observed) |
| Resampling | 2:5 (linear) |
| Sliding windows | Enabled, strict, step 25 |
| Screening | Prune agents with fewer than 2 observations |
| Lane-change sampling | Require 3 lane changes; keep 1 in 3 negatives |
| Maps | Full map |

## Dataset compatibility

Expected raw data layout for this loader.

| Field | Value |
| ----- | ----- |
| Expected release/layout | exiD v2.1 |
| Loader expectation | The loader assumes the exiD v2.1 distribution layout. |

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

Use the CLI for current native split and assignment support.

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
