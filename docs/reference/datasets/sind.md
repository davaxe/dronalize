# SIND

<div class="section-intro" markdown="1">
SIND is a drone dataset for signalized intersections in China. It combines multi-agent urban motion with traffic-light information, which makes it particularly useful for studying interaction under explicit traffic control.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Urban</strong></div>
  <div class="summary-item"><span>Release year</span><strong>2022</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Mixed</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Drone + camera</strong></div>
  <div class="summary-item"><span>Map context</span><strong>HD</strong></div>
  <div class="summary-item"><span># Samples</span><strong>Processed samples planned</strong></div>
</div>

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

## Dataset compatibility

Dronalize targets the release or raw layout below. If you have an older or newer download, expect breakage when split names, file names, schemas, or map assets differ.

| Field | Value |
| ----- | ----- |
| Expected release/layout | SinD city and recording directory layout |
| Loader expectation | The loader uses the city and recording directories directly and does not parse a separate release marker. |

## Normalization

### Agent categories

| Dataset type | Dronalize type |
| ------------ | -------------- |
| `motorcycle` | `MOTORCYCLE` |
| `car` | `CAR` |
| `truck` | `TRUCK` |
| `tricycle` | `TRICYCLE` |
| `bus` | `BUS` |
| `bicycle` | `BICYCLE` |
| `pedestrian` | `PEDESTRIAN` |
| `animal` | `ANIMAL` |

### Map types

| Dataset type | Dronalize type |
| ------------ | -------------- |
| `road_border` | `ROAD_BORDER` |
| `fence` | `ROAD_BORDER` |
| `wall` | `ROAD_BORDER` |
| `curbstone` | `CURB` |
| `stop_line` | `STOP` |
| `regulatory_element` | `REGULATORY` |
| `virtual` | `VIRTUAL` |
| `pedestrian_marking` | `PEDESTRIAN_MARKING` |
| `bike_marking` | `BIKE_MARKING` |
| `guard_rail` | `GUARD_RAIL` |
| `line_thin` with `subtype=dashed` | `LINE_THIN_DASHED` |
| `line_thin` without `subtype=dashed` | `LINE_THIN` |
| `line_thick` with `subtype=dashed` | `LINE_THICK_DASHED` |
| `line_thick` without `subtype=dashed` | `LINE_THICK` |

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
