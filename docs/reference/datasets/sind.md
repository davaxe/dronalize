# SIND

<div class="section-intro" markdown="1">
SIND is a drone dataset for signalized intersections in China. It combines multi-agent urban motion with traffic-light information, which makes it particularly useful for studying interaction under explicit traffic control.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Urban</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Mixed</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Drone + camera</strong></div>
  <div class="summary-item"><span>Map context</span><strong>HD</strong></div>
  <div class="summary-item"><span># Samples</span><strong>Processed samples planned</strong></div>
</div>

## Dataset facts

| Field               | Value                                                              | Notes                                                              |
| ------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| Release year        | 2022                                                               | Based on the cited dataset paper and release.                      |
| Domain              | Signalized urban intersections                                     | Designed for controlled intersection behavior and interaction.     |
| Capture platform    | Drone                                                              | Recorded from an overhead aerial perspective.                      |
| Primary agent types | Cars, trucks, buses, motorcycles, tricycles, bicycles, pedestrians | One of the broader road-user mixes among drone datasets.           |
| Map context         | HD-style road layout plus signal context                           | The signalized setting is central to the benchmark.                |
| Geographic coverage | China                                                              | Recorded at four urban intersections.                              |
| Data format         | Per-site trajectory folders with map files                         | Data and map assets are distributed separately within the release. |

## Default processing profile

These are the default Dronalize settings used when processing this dataset.

| Setting | Default |
| ------- | ------- |
| Source sequence | 20 obs / 50 pred @ 10 Hz |
| Effective sequence | 20 obs / 50 pred @ 10 Hz |
| Resampling | None |
| Windowing | 70-frame window, step 25 |
| Filtering | Prune agents with fewer than 2 samples |
| Maps | Full map |

## Version

Dronalize does not currently rely on a stable release version marker for SinD. The inspected raw layout does not expose a dataset version beyond the city and recording directories.

## Normalization

### Agent categories

| Dataset type | Dronalize type | Notes |
| ------------ | -------------- | ----- |
| `motorcycle` | `MOTORCYCLE` | Vehicle-track category mapping from `Veh_smoothed_tracks.csv`. |
| `car` | `CAR` | Vehicle-track category mapping from `Veh_smoothed_tracks.csv`. |
| `truck` | `TRUCK` | Vehicle-track category mapping from `Veh_smoothed_tracks.csv`. |
| `tricycle` | `TRICYCLE` | Vehicle-track category mapping from `Veh_smoothed_tracks.csv`. |
| `bus` | `BUS` | Vehicle-track category mapping from `Veh_smoothed_tracks.csv`. |
| `bicycle` | `BICYCLE` | Vehicle-track category mapping from `Veh_smoothed_tracks.csv`. |
| `pedestrian` | `PEDESTRIAN` | Pedestrian-track category mapping from `Ped_smoothed_tracks.csv`. |
| `animal` | `ANIMAL` | Pedestrian-track category mapping from `Ped_smoothed_tracks.csv`. |

### Map types

| Dataset type | Dronalize type | Notes |
| ------------ | -------------- | ----- |
| `road_border` | `ROAD_BORDER` | Shared Lanelet2/OSM mapping after SinD-specific coordinate recentering. |
| `fence` | `ROAD_BORDER` | Shared Lanelet2/OSM mapping after SinD-specific coordinate recentering. |
| `wall` | `ROAD_BORDER` | Shared Lanelet2/OSM mapping after SinD-specific coordinate recentering. |
| `curbstone` | `CURB` | Shared Lanelet2/OSM mapping after SinD-specific coordinate recentering. |
| `stop_line` | `STOP` | Shared Lanelet2/OSM mapping after SinD-specific coordinate recentering. |
| `regulatory_element` | `REGULATORY` | Shared Lanelet2/OSM mapping after SinD-specific coordinate recentering. |
| `virtual` | `VIRTUAL` | Shared Lanelet2/OSM mapping after SinD-specific coordinate recentering. |
| `pedestrian_marking` | `PEDESTRIAN_MARKING` | Shared Lanelet2/OSM mapping after SinD-specific coordinate recentering. |
| `bike_marking` | `BIKE_MARKING` | Shared Lanelet2/OSM mapping after SinD-specific coordinate recentering. |
| `guard_rail` | `GUARD_RAIL` | Shared Lanelet2/OSM mapping after SinD-specific coordinate recentering. |
| `line_thin` with `subtype=dashed` | `LINE_THIN_DASHED` | Shared Lanelet2/OSM mapping after SinD-specific coordinate recentering. |
| `line_thin` without `subtype=dashed` | `LINE_THIN` | Shared Lanelet2/OSM mapping after SinD-specific coordinate recentering. |
| `line_thick` with `subtype=dashed` | `LINE_THICK_DASHED` | Shared Lanelet2/OSM mapping after SinD-specific coordinate recentering. |
| `line_thick` without `subtype=dashed` | `LINE_THICK` | Shared Lanelet2/OSM mapping after SinD-specific coordinate recentering. |

## Split support

Use the command below for the most up-to-date split support information for this dataset, including native splits, supported custom split strategies, and any recommended strategy.

```bash
dronalize split-support sind
```

## References

- Dataset paper: [SIND: A Drone Dataset at Signalized Intersection in China](https://arxiv.org/abs/2209.02297)

## Expected structure

```text
sind/
├── Chanchun/
│   ├── changchun_pudong_507_009/
│   ├── changchun_pudong_507_010/
│   ├── ...
│   └── Chanchun_Pudom.osm
├── Chongqing/
│   ├── 6_22_NR_1/
│   ├── 6_22_NR_2/
│   ├── ...
│   └── NR_ll2.osm
├── Tianjin/
│   ├── 7_28_1/
│   ├── 8_2_1/
│   ├── ...
│   └── map_relink_law_save.osm
└── Xi'an/
    ├── xian_412_m1/
    ├── xian_412_m2/
    ├── ...
    └── Xi'an_Shanglin.osm


```
