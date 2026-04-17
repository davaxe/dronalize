# highD

<div class="section-intro" markdown="1">
highD is a widely used naturalistic highway trajectory dataset collected from drones over German freeways. It became a foundational benchmark for lane-level traffic analysis, safety validation, and highway motion prediction.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Highway</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Vehicles</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Drone</strong></div>
  <div class="summary-item"><span>Map context</span><strong>Limited</strong></div>
  <div class="summary-item"><span># Samples</span><strong>Processed samples planned</strong></div>
</div>

## Dataset facts

| Field               | Value                       | Notes                                                                   |
| ------------------- | --------------------------- | ----------------------------------------------------------------------- |
| Release year        | 2018                        | Based on the cited dataset paper and release.                           |
| Domain              | Highway traffic             | Focused on naturalistic freeway driving.                                |
| Capture platform    | Drone                       | Overhead aerial view reduces occlusion and preserves lane-level motion. |
| Primary agent types | Cars and trucks             | Built around motorized highway traffic.                                 |
| Map context         | Lane-level roadway geometry | Well suited for lane-change and safety studies.                         |
| Geographic coverage | German highways             | Covers multiple recording sites rather than one corridor.               |
| Data format         | CSV trajectory files        | Organized as recording metadata, tracks, and track metadata.            |

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

Dronalize does not currently rely on a stable release version marker for highD. The inspected raw layout does not expose a dataset version beyond the recording bundle itself.

## Normalization

### Agent categories

| Dataset type | Dronalize type | Notes |
| ------------ | -------------- | ----- |
| `Car` | `CAR` | Direct category mapping from `class`. |
| `Truck` | `TRUCK` | Direct category mapping from `class`. |

### Map types

| Dataset type | Dronalize type | Notes |
| ------------ | -------------- | ----- |
| First or last entry in `upperLaneMarkings` | `ROAD_BORDER` | The outermost upper-carriageway markings are treated as road borders. |
| Interior entry in `upperLaneMarkings` | `LINE_THIN_DASHED` | Intermediate upper-carriageway markings are treated as dashed lane separators. |
| First or last entry in `lowerLaneMarkings` | `ROAD_BORDER` | The outermost lower-carriageway markings are treated as road borders. |
| Interior entry in `lowerLaneMarkings` | `LINE_THIN_DASHED` | Intermediate lower-carriageway markings are treated as dashed lane separators. |

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
