# VoD

<div class="section-intro" markdown="1">
The View-of-Delft prediction dataset is an urban mixed-traffic benchmark with a comparatively strong vulnerable-road-user presence. It is useful when you want an autonomous-driving prediction dataset that is less vehicle-dominated than many mainstream benchmarks.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Urban</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Mixed</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Vehicle</strong></div>
  <div class="summary-item"><span>Map context</span><strong>HD</strong></div>
  <div class="summary-item"><span># Samples</span><strong>Processed samples planned</strong></div>
</div>

## Dataset facts

| Field               | Value                           | Notes                                                                                            |
| ------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------ |
| Release year        | 2024                            | Based on the cited dataset paper and release.                                                   |
| Domain              | Urban mixed traffic             | Focused on dense city traffic with more vulnerable road users than many self-driving benchmarks. |
| Capture platform    | Processed benchmark release     | Released as scene metadata and map assets.                                                       |
| Primary agent types | Vehicles, pedestrians, cyclists | Explicitly framed as a multi-class prediction dataset.                                           |
| Map context         | HD semantic maps                | Includes semantic map material for the Delft environment.                                        |
| Geographic coverage | Delft, Netherlands              | Concentrated in one urban area.                                                                  |
| Data format         | Scene metadata plus map assets  | Structured into train/validation and test releases.                                              |

## Default processing profile

These are the default Dronalize settings used when processing this dataset.

| Setting | Default |
| ------- | ------- |
| Source sequence | 5 obs / 30 pred @ 10 Hz |
| Effective sequence | 5 obs / 30 pred @ 10 Hz |
| Resampling | None |
| Windowing | 35-frame window, step 5 |
| Filtering | Drop parked and undefined actors, remove duplicate `vehicle.ego` instances, and prune agents with fewer than 2 samples |
| Maps | Relevant area (padding 1.15) |

## Version

Dronalize currently targets VoD `v1.0`, matching the inspected `v1.0-trainval` and `v1.0-test` layout.

## Normalization

### Agent categories

| Dataset type | Dronalize type | Notes |
| ------------ | -------------- | ----- |
| `vehicle.car` | `CAR` | VoD reuses the nuScenes category mapping. |
| `vehicle.ego.car` | `CAR` | VoD reuses the nuScenes category mapping; duplicate `vehicle.ego` rows are filtered by default loader options. |
| `vehicle.van` | `VAN` | VoD reuses the nuScenes category mapping. |
| `vehicle.construction` | `VAN` | VoD reuses the nuScenes category mapping. |
| `vehicle.bus.bendy` | `BUS` | VoD reuses the nuScenes category mapping. |
| `vehicle.bus.rigid` | `BUS` | VoD reuses the nuScenes category mapping. |
| `vehicle.truck` | `TRUCK` | VoD reuses the nuScenes category mapping. |
| `vehicle.trailer` | `TRAILER` | VoD reuses the nuScenes category mapping. |
| `vehicle.motorcycle` | `MOTORCYCLE` | VoD reuses the nuScenes category mapping. |
| `vehicle.bicycle` | `BICYCLE` | VoD reuses the nuScenes category mapping. |
| `vehicle.emergency.ambulance` | `CAR` | VoD reuses the nuScenes category mapping. |
| `vehicle.emergency.police` | `CAR` | VoD reuses the nuScenes category mapping. |
| `human.pedestrian.adult` | `PEDESTRIAN` | VoD reuses the nuScenes category mapping. |
| `human.pedestrian.child` | `PEDESTRIAN` | VoD reuses the nuScenes category mapping. |
| `human.pedestrian.construction_worker` | `PEDESTRIAN` | VoD reuses the nuScenes category mapping. |
| `human.pedestrian.police_officer` | `PEDESTRIAN` | VoD reuses the nuScenes category mapping. |
| `human.pedestrian.stroller` | `PEDESTRIAN` | VoD reuses the nuScenes category mapping. |
| `human.pedestrian.wheelchair` | `PEDESTRIAN` | VoD reuses the nuScenes category mapping. |
| `human.pedestrian` | `PEDESTRIAN` | VoD reuses the nuScenes category mapping. |
| `static_object.bicycle_rack` | `STATIC_OBJECT` | VoD reuses the nuScenes category mapping. |
| `movable_object.barrier` | `MOVEABLE_OBJECT` | VoD reuses the nuScenes category mapping. |
| `movable_object.debris` | `MOVEABLE_OBJECT` | VoD reuses the nuScenes category mapping. |
| `movable_object.pushable_pullable` | `MOVEABLE_OBJECT` | VoD reuses the nuScenes category mapping. |
| `movable_object.trafficcone` | `MOVEABLE_OBJECT` | VoD reuses the nuScenes category mapping. |
| `animal` | `ANIMAL` | VoD reuses the nuScenes category mapping. |

### Map types

| Dataset type | Dronalize type | Notes |
| ------------ | -------------- | ----- |
| `SegmentDividerType.NIL` | `VIRTUAL` | VoD reuses the nuScenes divider mapping. |
| `SegmentDividerType.SINGLE_SOLID_WHITE` | `LINE_THIN` | VoD reuses the nuScenes divider mapping. |
| `SegmentDividerType.SINGLE_SOLID_YELLOW` | `LINE_THIN` | VoD reuses the nuScenes divider mapping. |
| `SegmentDividerType.SINGLE_ZIGZAG_WHITE` | `REGULATORY` | VoD reuses the nuScenes divider mapping. |
| `SegmentDividerType.DOUBLE_SOLID_WHITE` | `LINE_THIN_DOUBLE` | VoD reuses the nuScenes divider mapping. |
| `SegmentDividerType.DOUBLE_DASHED_WHITE` | `LINE_THIN_DOUBLE_DASHED` | VoD reuses the nuScenes divider mapping. |
| Lane polygon outline | `LINE_THIN` | Unlike nuScenes, the VoD builder sets `lane_polygon_edge=LINE_THIN`. |

## Split support

Use the command below for the most up-to-date split support information for this dataset, including native splits, supported custom split strategies, and any recommended strategy.

```bash
dronalize split-support vod
```

## References

- Dataset paper: [Multi-class Trajectory Prediction in Urban Traffic using the View-of-Delft Prediction Dataset](https://pure.tudelft.nl/ws/portalfiles/portal/190220102/Multi-Class_Trajectory_Prediction_in_Urban_Traffic_Using_the_View-of-Delft_Prediction_Dataset.pdf)

## Expected structure

```text
vod/
├── maps/
│   └── expansion/
│       └── delft.json
├── v1.0-trainval/
└── v1.0-test/
```
