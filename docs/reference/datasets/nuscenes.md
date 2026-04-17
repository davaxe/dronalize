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
| Filtering | Drop parked and undefined actors, ignore categories matching `object`, and prune agents with fewer than 2 samples |
| Maps | Relevant area (padding 1.15) |

## Version

Dronalize currently targets nuScenes `v1.0` metadata together with the `v1.3` map expansion, matching the inspected `v1.0-trainval_meta`, `v1.0-test_meta`, and `nuScenes-map-expansion-v1.3` layout.

## Normalization

### Agent categories

| Dataset type | Dronalize type | Notes |
| ------------ | -------------- | ----- |
| `vehicle.car` | `CAR` | nuScenes category mapping in `_FULL_CATEGORY_MAPPING`. |
| `vehicle.ego.car` | `CAR` | nuScenes category mapping in `_FULL_CATEGORY_MAPPING`. |
| `vehicle.van` | `VAN` | nuScenes category mapping in `_FULL_CATEGORY_MAPPING`. |
| `vehicle.construction` | `VAN` | Construction vehicles are merged into the shared van category. |
| `vehicle.bus.bendy` | `BUS` | nuScenes category mapping in `_FULL_CATEGORY_MAPPING`. |
| `vehicle.bus.rigid` | `BUS` | nuScenes category mapping in `_FULL_CATEGORY_MAPPING`. |
| `vehicle.truck` | `TRUCK` | nuScenes category mapping in `_FULL_CATEGORY_MAPPING`. |
| `vehicle.trailer` | `TRAILER` | nuScenes category mapping in `_FULL_CATEGORY_MAPPING`. |
| `vehicle.motorcycle` | `MOTORCYCLE` | nuScenes category mapping in `_FULL_CATEGORY_MAPPING`. |
| `vehicle.bicycle` | `BICYCLE` | nuScenes category mapping in `_FULL_CATEGORY_MAPPING`. |
| `vehicle.emergency.ambulance` | `CAR` | Emergency road vehicles are currently merged into the shared car category. |
| `vehicle.emergency.police` | `CAR` | Emergency road vehicles are currently merged into the shared car category. |
| `human.pedestrian.adult` | `PEDESTRIAN` | nuScenes category mapping in `_FULL_CATEGORY_MAPPING`. |
| `human.pedestrian.child` | `PEDESTRIAN` | nuScenes category mapping in `_FULL_CATEGORY_MAPPING`. |
| `human.pedestrian.construction_worker` | `PEDESTRIAN` | nuScenes category mapping in `_FULL_CATEGORY_MAPPING`. |
| `human.pedestrian.police_officer` | `PEDESTRIAN` | nuScenes category mapping in `_FULL_CATEGORY_MAPPING`. |
| `human.pedestrian.stroller` | `PEDESTRIAN` | nuScenes category mapping in `_FULL_CATEGORY_MAPPING`. |
| `human.pedestrian.wheelchair` | `PEDESTRIAN` | nuScenes category mapping in `_FULL_CATEGORY_MAPPING`. |
| `human.pedestrian` | `PEDESTRIAN` | nuScenes category mapping in `_FULL_CATEGORY_MAPPING`. |
| `static_object.bicycle_rack` | `STATIC_OBJECT` | nuScenes category mapping in `_FULL_CATEGORY_MAPPING`. |
| `movable_object.barrier` | `MOVEABLE_OBJECT` | nuScenes category mapping in `_FULL_CATEGORY_MAPPING`. |
| `movable_object.debris` | `MOVEABLE_OBJECT` | nuScenes category mapping in `_FULL_CATEGORY_MAPPING`. |
| `movable_object.pushable_pullable` | `MOVEABLE_OBJECT` | nuScenes category mapping in `_FULL_CATEGORY_MAPPING`. |
| `movable_object.trafficcone` | `MOVEABLE_OBJECT` | nuScenes category mapping in `_FULL_CATEGORY_MAPPING`. |
| `animal` | `ANIMAL` | nuScenes category mapping in `_FULL_CATEGORY_MAPPING`. |

### Map types

| Dataset type | Dronalize type | Notes |
| ------------ | -------------- | ----- |
| `SegmentDividerType.NIL` | `VIRTUAL` | nuScenes divider mapping in the parser. |
| `SegmentDividerType.SINGLE_SOLID_WHITE` | `LINE_THIN` | nuScenes divider mapping in the parser. |
| `SegmentDividerType.SINGLE_SOLID_YELLOW` | `LINE_THIN` | nuScenes divider mapping in the parser. |
| `SegmentDividerType.SINGLE_ZIGZAG_WHITE` | `REGULATORY` | nuScenes divider mapping in the parser. |
| `SegmentDividerType.DOUBLE_SOLID_WHITE` | `LINE_THIN_DOUBLE` | nuScenes divider mapping in the parser. |
| `SegmentDividerType.DOUBLE_DASHED_WHITE` | `LINE_THIN_DOUBLE_DASHED` | nuScenes divider mapping in the parser. |
| Lane polygon outline | Not emitted | The current nuScenes builder keeps `lane_polygon_edge=None`. |

## Split support

Use the command below for the most up-to-date split support information for this dataset, including native splits, supported custom split strategies, and any recommended strategy.

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
