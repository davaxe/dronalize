# OpenDD

OpenDD is a large-scale drone dataset for roundabout traffic. It combines many tracked trajectories with detailed roundabout context and is a strong benchmark for dense interaction analysis in circular junctions.

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Roundabout</strong></div>
  <div class="summary-item"><span>Release year</span><strong>2020</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Mixed</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Drone</strong></div>
  <div class="summary-item"><span>Map context</span><strong>HD</strong></div>
  <div class="summary-item"><span># Records</span><strong>processed scene records planned</strong></div>
</div>

## Default prediction task

The `benchmark` prediction task is selected automatically. In the project config, select it explicitly with `task = "benchmark"` or disable prediction bounds with `task = "none"`.

| Setting | Default |
| ------- | ------- |
| Configured horizon | 210 frames (benchmark origin 60) @ 30.0 Hz |
| Effective horizon | 70 frames (benchmark origin 20) @ 10.0 Hz |
| Source unit | Recording |
| Source bounds | 456-16916 frames (observed) |
| Resampling | 1:3 (linear) |
| Sliding windows | Enabled, strict, step 75 |
| Screening | Prune agents with fewer than 6 observations |
| Maps | Full map |

## Dataset compatibility

Expected raw data layout for this loader.

| Field | Value |
| ----- | ----- |
| Expected release/layout | OpenDD v3 SQLite layout |
| Loader expectation | The loader currently scans `trajectories_*_v3.sqlite`, so other OpenDD layouts will not match automatically. |

## Normalization

### Agent categories

| Dataset type | Prejectory type |
| ------------ | -------------- |
| `Car` | `CAR` |
| `Medium Vehicle` | `CAR` |
| `Heavy Vehicle` | `TRUCK` |
| `Trailer` | `TRUCK` |
| `Bus` | `BUS` |
| `Motorcycle` | `MOTORCYCLE` |
| `Pedestrian` | `PEDESTRIAN` |
| `Bicycle` | `BICYCLE` |

### Map types

| Dataset type | Prejectory type |
| ------------ | -------------- |
| `CURB` | `CURB` |
| `CURB_TRAVERSABLE` | `CURB` |
| `SHORT_DASHED_LINE` | `LINE_THIN_DASHED` |
| `LONG_DASHED_LINE` | `LINE_THIN_DASHED` |
| `SINGLE_SOLID_LINE` | `LINE_THIN` |
| `NO_MARKING` | `VIRTUAL` |
| `SHADED_AREA_MARKING` | `VIRTUAL` |
| `GUARDRAIL` | `GUARD_RAIL` |

## Split support

Use the CLI for current native split and assignment support.

```bash
prejectory split-support opendd
```

## References

- Dataset paper: [openDD: A large-scale roundabout drone dataset](https://arxiv.org/abs/2007.08463)

## Expected structure

```text
opendd/
├── opendd_v3-rdb1/
│   ├── rdb1/
│   │   └── map_rdb1/
│   └── trajectories_rdb1_v3.sqlite
├── opendd_v3-rdb2/
└── opendd_v3-rdb7/
```
