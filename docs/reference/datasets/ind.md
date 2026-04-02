# inD

<div class="section-intro" markdown="1">
inD is a naturalistic urban-intersection dataset captured from drones above German intersections. It extends the drone-dataset style beyond highways and emphasizes multimodal interaction among vehicles, cyclists, and pedestrians.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Urban</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Mixed</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Drone</strong></div>
  <div class="summary-item"><span>Map context</span><strong>HD</strong></div>
  <div class="summary-item"><span># Samples</span><strong>Processed samples planned</strong></div>
</div>

## Dataset facts

| Field               | Value                                        | Notes                                                    |
| ------------------- | -------------------------------------------- | -------------------------------------------------------- |
| Release year        | 2020                                         | Based on the cited dataset paper and release.            |
| Domain              | Urban intersections                          | Designed for interaction-heavy city traffic.             |
| Capture platform    | Drone                                        | Recorded from an overhead aerial viewpoint.              |
| Primary agent types | Cars, trucks or buses, bicycles, pedestrians | Stronger multimodal focus than highway-only benchmarks.  |
| Map context         | Intersection road layout                     | Includes map assets for the recorded sites.              |
| Geographic coverage | Germany                                      | Spans four urban intersections.                          |
| Data format         | CSV trajectories with maps                   | Organized as recording metadata, tracks, and map assets. |

## Default processing profile

These are the default Dronalize settings used when processing this dataset.

| Setting | Default |
| ------- | ------- |
| Source sequence | 50 obs / 125 pred @ 25 Hz |
| Effective sequence | 99 obs / 250 pred @ 50 Hz |
| Resampling | Cubic 2:1 |
| Windowing | 175-frame window, step 25 |
| Filtering | Exclude trailers; prune agents with fewer than 6 samples |
| Maps | Full map |

### Filtering details

| Scope | Rule | Effect |
| ----- | ---- | ------ |
| Cleanup | Exclude categories | Remove trailer tracks. |
| Cleanup | Minimum samples | Prune agents with fewer than 6 samples. |

## Split support

Use the command below for the most up-to-date split support information for this dataset, including native splits, supported custom modes, and any recommended strategy.

```bash
dronalize split-support ind
```

## References

- Dataset paper: [The inD Dataset: A Drone Dataset of Naturalistic Road User Trajectories at German Intersections](https://arxiv.org/abs/1911.07602)

## Expected structure

```text
inD/
├── data/
│   ├── 01_recordingMeta.csv
│   ├── 01_tracks.csv
│   ├── 01_tracksMeta.csv
│   └── ...
└── maps/
    └── ...
```
