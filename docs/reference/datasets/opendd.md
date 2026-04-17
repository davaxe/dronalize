# OpenDD

<div class="section-intro" markdown="1">
OpenDD is a large-scale drone dataset for roundabout traffic. It combines many tracked trajectories with detailed roundabout context and is a strong benchmark for dense interaction analysis in circular junctions.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Roundabout</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Mixed</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Drone</strong></div>
  <div class="summary-item"><span>Map context</span><strong>HD</strong></div>
  <div class="summary-item"><span># Samples</span><strong>Processed samples planned</strong></div>
</div>

## Dataset facts

| Field               | Value                                      | Notes                                                                   |
| ------------------- | ------------------------------------------ | ----------------------------------------------------------------------- |
| Release year        | 2020                                       | Based on the cited dataset paper and release.                           |
| Domain              | Roundabout traffic                         | Focused on circular-junction interaction.                               |
| Capture platform    | Drone                                      | Overhead recording supports wide scene coverage with limited occlusion. |
| Primary agent types | Cars, bicycles, pedestrians, trucks, buses | Covers both motorized and vulnerable road users.                        |
| Map context         | Roundabout road geometry                   | Includes map material for the recorded sites.                           |
| Geographic coverage | Germany                                    | Recorded across seven German roundabouts.                               |
| Data format         | SQLite trajectories plus map files         | Each roundabout release contains trajectory and map assets.             |

## Default processing profile

These are the default Dronalize settings used when processing this dataset.

| Setting | Default |
| ------- | ------- |
| Source sequence | 60 obs / 150 pred @ 30 Hz |
| Effective sequence | 20 obs / 50 pred @ 10 Hz |
| Resampling | Linear 1:3 |
| Windowing | 210-frame window, step 75 |
| Filtering | Prune agents with fewer than 6 samples |
| Maps | Full map |

## Version

Dronalize currently targets the OpenDD `v3` SQLite layout, matching both the loader glob `trajectories_*_v3.sqlite` and the inspected `opendd_v3-rdb*` directories.

## Normalization

### Agent categories

| Dataset type | Dronalize type | Notes |
| ------------ | -------------- | ----- |
| `Car` | `CAR` | Direct category mapping from `CLASS`. |
| `Medium Vehicle` | `CAR` | Medium vehicles are merged into the car category. |
| `Heavy Vehicle` | `TRUCK` | Heavy vehicles are merged into the truck category. |
| `Trailer` | `TRUCK` | Trailers are merged into the truck category. |
| `Bus` | `BUS` | Direct category mapping from `CLASS`. |
| `Motorcycle` | `MOTORCYCLE` | Direct category mapping from `CLASS`. |
| `Pedestrian` | `PEDESTRIAN` | Direct category mapping from `CLASS`. |
| `Bicycle` | `BICYCLE` | Direct category mapping from `CLASS`. |

### Map types

| Dataset type | Dronalize type | Notes |
| ------------ | -------------- | ----- |
| `CURB` | `CURB` | Direct map-marking mapping from the OpenDD SQLite map tables. |
| `CURB_TRAVERSABLE` | `CURB` | Traversable curbs are merged into the curb category. |
| `SHORT_DASHED_LINE` | `LINE_THIN_DASHED` | Direct map-marking mapping from the OpenDD SQLite map tables. |
| `LONG_DASHED_LINE` | `LINE_THIN_DASHED` | Direct map-marking mapping from the OpenDD SQLite map tables. |
| `SINGLE_SOLID_LINE` | `LINE_THIN` | Direct map-marking mapping from the OpenDD SQLite map tables. |
| `NO_MARKING` | `VIRTUAL` | Used for boundaries that should not render as painted or physical edges. |
| `SHADED_AREA_MARKING` | `VIRTUAL` | Used for shaded areas that should not render as painted or physical edges. |
| `GUARDRAIL` | `GUARD_RAIL` | Direct map-marking mapping from the OpenDD SQLite map tables. |

## Split support

Use the command below for the most up-to-date split support information for this dataset, including native splits, supported custom split strategies, and any recommended strategy.

```bash
dronalize split-support opendd
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
