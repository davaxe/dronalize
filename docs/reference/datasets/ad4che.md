# AD4CHE

<div class="section-intro" markdown="1">
AD4CHE is an aerial congestion dataset for highway and expressway traffic in China. It is aimed at interaction-heavy congestion scenarios, especially the kinds of cut-ins and traffic-jam behavior that matter for assisted-driving evaluation.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Highway traffic</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Vehicles</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Drone</strong></div>
  <div class="summary-item"><span>Map context</span><strong>Limited</strong></div>
  <div class="summary-item"><span># Samples</span><strong>Processed samples planned</strong></div>
</div>

## Dataset facts

| Field               | Value                             | Notes                                                                     |
| ------------------- | --------------------------------- | ------------------------------------------------------------------------- |
| Release year        | 2023                              | Based on the cited dataset paper and release.                             |
| Domain              | Congested highway traffic         | Focused on highways and expressways in urban Chinese settings.            |
| Capture platform    | Drone                             | Recorded from an overhead aerial perspective.                             |
| Primary agent types | Cars, trucks, buses, motorcycles  | Covers mixed motorized traffic in congestion.                             |
| Map context         | Lane imagery                      | Includes lane images that provide road context for each recording.        |
| Geographic coverage | Four cities in China              | Designed to span multiple congestion scenarios rather than a single site. |
| Data format         | CSV trajectories with lane images | Trajectories are paired with recording metadata and lane-view assets.     |

## Default processing profile

These are the default Dronalize settings used when processing this dataset.

| Setting | Default |
| ------- | ------- |
| Source sequence | 60 obs / 150 pred @ 30 Hz |
| Effective sequence | 20 obs / 50 pred @ 10 Hz |
| Resampling | Linear 1:3 |
| Windowing | 210-frame window, step 45 |
| Filtering | Drop short tracks with fewer than 4 samples |
| Lane-change sampling | Require 5 lane changes; keep 1 in 3 negative scenes |
| Maps | Relevant area (padding 1.15) |

## Split support

Use the command below for the most up-to-date split support information for this dataset, including native splits, supported custom split strategies, and any recommended strategy.

```bash
dronalize split-support ad4che
```

## References

- Dataset paper: [The AD4CHE dataset and its application in typical congestion scenarios of traffic jam pilot systems](https://ieeexplore.ieee.org/document/10079130)

## Expected structure

```text
ad4che/
└── AD4CHE_Data_V1.0/
    ├── DJI_0001/
    │   ├── 01_lanePicture.png
    │   ├── 01_recordingMeta.csv
    │   ├── 01_tracksMeta.csv
    │   └── 01_tracks.csv
    ├── DJI_0002/
    └── ...
```

## Notes

- AD4CHE represents road context through lane images rather than a conventional benchmark map package.
