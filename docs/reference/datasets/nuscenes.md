# nuScenes

<div class="section-intro" markdown="1">
nuScenes is a multimodal autonomous-driving benchmark that combines tracked actors, sensor data, and city-scale map context. It is one of the most widely used general-purpose datasets for self-driving perception and forecasting research.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Mixed urban</strong></div>
  <div class="summary-item"><span>Release year</span><strong>2020</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Mixed</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Vehicle</strong></div>
  <div class="summary-item"><span>Map context</span><strong>HD</strong></div>
  <div class="summary-item"><span># Samples</span><strong>Processed samples planned</strong></div>
</div>

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

## Dataset compatibility

Dronalize targets the release or raw layout below. If you have an older or newer download, expect breakage when split names, file names, schemas, or map assets differ.

| Field | Value |
| ----- | ----- |
| Expected release/layout | nuScenes v1.0 metadata with map expansion v1.3 |
| Loader expectation | The loader expects `v1.0-trainval_meta`, `v1.0-test_meta`, and `nuScenes-map-expansion-v1.3`. |

## Normalization

### Agent categories

| Dataset type | Dronalize type |
| ------------ | -------------- |
| `vehicle.car` | `CAR` |
| `vehicle.ego.car` | `CAR` |
| `vehicle.van` | `VAN` |
| `vehicle.construction` | `VAN` |
| `vehicle.bus.bendy` | `BUS` |
| `vehicle.bus.rigid` | `BUS` |
| `vehicle.truck` | `TRUCK` |
| `vehicle.trailer` | `TRAILER` |
| `vehicle.motorcycle` | `MOTORCYCLE` |
| `vehicle.bicycle` | `BICYCLE` |
| `vehicle.emergency.ambulance` | `CAR` |
| `vehicle.emergency.police` | `CAR` |
| `human.pedestrian.adult` | `PEDESTRIAN` |
| `human.pedestrian.child` | `PEDESTRIAN` |
| `human.pedestrian.construction_worker` | `PEDESTRIAN` |
| `human.pedestrian.police_officer` | `PEDESTRIAN` |
| `human.pedestrian.stroller` | `PEDESTRIAN` |
| `human.pedestrian.wheelchair` | `PEDESTRIAN` |
| `human.pedestrian` | `PEDESTRIAN` |
| `static_object.bicycle_rack` | `STATIC_OBJECT` |
| `movable_object.barrier` | `MOVEABLE_OBJECT` |
| `movable_object.debris` | `MOVEABLE_OBJECT` |
| `movable_object.pushable_pullable` | `MOVEABLE_OBJECT` |
| `movable_object.trafficcone` | `MOVEABLE_OBJECT` |
| `animal` | `ANIMAL` |

### Map types

| Dataset type | Dronalize type |
| ------------ | -------------- |
| `SegmentDividerType.NIL` | `VIRTUAL` |
| `SegmentDividerType.SINGLE_SOLID_WHITE` | `LINE_THIN` |
| `SegmentDividerType.SINGLE_SOLID_YELLOW` | `LINE_THIN` |
| `SegmentDividerType.SINGLE_ZIGZAG_WHITE` | `REGULATORY` |
| `SegmentDividerType.DOUBLE_SOLID_WHITE` | `LINE_THIN_DOUBLE` |
| `SegmentDividerType.DOUBLE_DASHED_WHITE` | `LINE_THIN_DOUBLE_DASHED` |
| Lane polygon outline | Not emitted |

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
└── v1.0-trainval_meta/
```
