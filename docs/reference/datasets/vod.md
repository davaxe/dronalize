# VoD

<div class="section-intro" markdown="1">
The View-of-Delft prediction dataset is an urban mixed-traffic benchmark with a comparatively strong vulnerable-road-user presence. It is useful when you want an autonomous-driving prediction dataset that is less vehicle-dominated than many mainstream benchmarks.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Urban</strong></div>
  <div class="summary-item"><span>Release year</span><strong>2024</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Mixed</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Vehicle</strong></div>
  <div class="summary-item"><span>Map context</span><strong>HD</strong></div>
  <div class="summary-item"><span># Samples</span><strong>Processed samples planned</strong></div>
</div>

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

## Dataset compatibility

Dronalize targets the release or raw layout below. If you have an older or newer download, expect breakage when split names, file names, schemas, or map assets differ.

| Field | Value |
| ----- | ----- |
| Expected release/layout | View of Delft v1.0 trainval and test layout |
| Loader expectation | The loader expects the `v1.0-trainval` and `v1.0-test` directory names. |

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
| Lane polygon outline | `LINE_THIN` |

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
