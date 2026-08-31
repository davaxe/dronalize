# highD

highD is a widely used naturalistic highway trajectory dataset collected from drones over German freeways. It became a foundational benchmark for lane-level traffic analysis, safety validation, and highway motion prediction.

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Highway</strong></div>
  <div class="summary-item"><span>Release year</span><strong>2018</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Vehicles</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Drone</strong></div>
  <div class="summary-item"><span>Map context</span><strong>Limited</strong></div>
  <div class="summary-item"><span># Records</span><strong>processed scene records planned</strong></div>
</div>

## Default prediction task

The `benchmark` prediction task is selected automatically. In the project config, select it explicitly with `task = "benchmark"` or disable prediction bounds with `task = "none"`.

| Setting | Default |
| ------- | ------- |
| Configured horizon | 175 frames (benchmark origin 50) @ 25.0 Hz |
| Effective horizon | 70 frames (benchmark origin 20) @ 10.0 Hz |
| Source unit | Recording |
| Source bounds | 9729-31274 frames (observed) |
| Resampling | 2:5 (linear) |
| Sliding windows | Enabled, strict, step 25 |
| Screening | Prune agents with fewer than 2 observations |
| Lane-change sampling | Require 3 lane changes; keep 1 in 3 negatives |
| Maps | Full map |

## Dataset compatibility

Expected raw data layout for this loader.

| Field | Value |
| ----- | ----- |
| Expected release/layout | highD recording bundle layout |
| Loader expectation | The loader uses the highD recording files directly and does not parse a stable upstream version marker. |

## Normalization

### Agent categories

| Dataset type | Prejectory type |
| ------------ | -------------- |
| `Car` | `CAR` |
| `Truck` | `TRUCK` |

### Map types

| Dataset type | Prejectory type |
| ------------ | -------------- |
| First or last entry in `upperLaneMarkings` | `ROAD_BORDER` |
| Interior entry in `upperLaneMarkings` | `LINE_THIN_DASHED` |
| First or last entry in `lowerLaneMarkings` | `ROAD_BORDER` |
| Interior entry in `lowerLaneMarkings` | `LINE_THIN_DASHED` |

## Split support

Use the CLI for current native split and assignment support.

```bash
prejectory split-support highd
```

## References

- Dataset paper: [The highD Dataset: A Drone Dataset of Naturalistic Vehicle Trajectories on German Highways for Validation of Highly Automated Driving Systems](https://arxiv.org/abs/1810.05642)

## Expected structure

```text
highd/
└── data/
    ├── 01_recordingMeta.csv
    ├── 01_tracks.csv
    ├── 01_tracksMeta.csv
    └── ...
```
