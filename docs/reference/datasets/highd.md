# highD

<div class="section-intro" markdown="1">
highD is a widely used naturalistic highway trajectory dataset collected from drones over German freeways. It became a foundational benchmark for lane-level traffic analysis, safety validation, and highway motion prediction.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Highway</strong></div>
  <div class="summary-item"><span>Release year</span><strong>2018</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Vehicles</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Drone</strong></div>
  <div class="summary-item"><span>Map context</span><strong>Limited</strong></div>
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
| Lane-change sampling | Require 3 lane changes; keep 1 in 3 negatives |
| Maps | Full map |

## Dataset compatibility

Dronalize targets the release or raw layout below. If you have an older or newer download, expect breakage when split names, file names, schemas, or map assets differ.

| Field | Value |
| ----- | ----- |
| Expected release/layout | highD recording bundle layout |
| Loader expectation | The loader uses the highD recording files directly and does not parse a stable upstream version marker. |

## Normalization

### Agent categories

| Dataset type | Dronalize type |
| ------------ | -------------- |
| `Car` | `CAR` |
| `Truck` | `TRUCK` |

### Map types

| Dataset type | Dronalize type |
| ------------ | -------------- |
| First or last entry in `upperLaneMarkings` | `ROAD_BORDER` |
| Interior entry in `upperLaneMarkings` | `LINE_THIN_DASHED` |
| First or last entry in `lowerLaneMarkings` | `ROAD_BORDER` |
| Interior entry in `lowerLaneMarkings` | `LINE_THIN_DASHED` |

## Split support

Use the command below for the most up-to-date split support information for this dataset, including native splits, supported custom split strategies, and any recommended strategy.

```bash
dronalize split-support highd
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
