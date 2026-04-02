# nuScenes

<div class="section-intro" markdown="1">
nuScenes is a multimodal autonomous-driving benchmark that combines tracked actors, sensor data, and city-scale map context. It is one of the most widely used general-purpose datasets for self-driving perception and forecasting research.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Mixed urban</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Mixed</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Vehicle</strong></div>
  <div class="summary-item"><span>Map context</span><strong>HD</strong></div>
  <div class="summary-item"><span># Samples</span><strong>Processed samples planned</strong></div>
</div>

## Dataset facts

| Field               | Value                              | Notes                                                     |
| ------------------- | ---------------------------------- | --------------------------------------------------------- |
| Release year        | 2020                               | Based on the cited dataset paper and benchmark release.   |
| Domain              | Mixed urban autonomous driving     | Used across perception, tracking, and forecasting tasks.  |
| Capture platform    | Self-driving vehicle fleet         | Includes camera, radar, lidar, and map assets.            |
| Primary agent types | Vehicles and vulnerable road users | Supports a broad set of traffic participants.             |
| Map context         | Map expansion files                | Rich road-layout context is part of the standard release. |
| Geographic coverage | Boston and Singapore               | Chosen to provide strong geographic diversity.            |
| Data format         | Metadata tables plus map assets    | Organized into train/validation and test releases.        |

## Default processing profile

These are the default Dronalize settings used when processing this dataset.

| Setting | Default |
| ------- | ------- |
| Source sequence | 4 obs / 12 pred @ 2 Hz |
| Effective sequence | 16 obs / 60 pred @ 10 Hz |
| Resampling | Linear 5:1 |
| Windowing | 16-frame window, step 1 |
| Filtering | Require last observation frame (3) |
| Maps | Full map |

## Split support

Use the command below for the most up-to-date split support information for this dataset, including native splits, supported custom modes, and any recommended strategy.

```bash
dronalize split-support nuscenes
```

## References

- Dataset paper: [nuScenes: A Multimodal Dataset for Autonomous Driving](https://arxiv.org/abs/1903.11027)

## Expected structure

```text
nuscenes/
├── nuScenes-map-expansion-v1.3/
│   └── expansion/
│       ├── boston-seaport.json
│       ├── singapore-onenorth.json
│       └── ...
├── v1.0-trainval_meta/
└── v1.0-test_meta/
```
