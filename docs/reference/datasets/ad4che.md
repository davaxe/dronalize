# AD4CHE

AD4CHE is an aerial congestion dataset for highway and expressway traffic in China. It is aimed at interaction-heavy congestion scenarios, especially the kinds of cut-ins and traffic-jam behavior that matter for assisted-driving evaluation.

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Highway traffic</strong></div>
  <div class="summary-item"><span>Release year</span><strong>2023</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Vehicles</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Drone</strong></div>
  <div class="summary-item"><span>Map context</span><strong>Limited</strong></div>
  <div class="summary-item"><span># Records</span><strong>processed scene records planned</strong></div>
</div>

!!! info "Extra dependencies"
    This dataset requires the `ad4che` extra to be installed. You can do this with pip (or your package manager of choice):

    ```bash
    pip install dronalize[ad4che]
    ```

## Default processing profile

Default processing settings for this dataset.

| Setting | Default |
| ------- | ------- |
| Configured horizon | 211 frames (default split after 61) @ 30.0 Hz |
| Effective horizon | 71 frames (default split after 21) @ 10.0 Hz |
| Source unit | Recording |
| Source bounds | 1136-9821 frames (observed) |
| Resampling | 1:3 (linear) |
| Sliding windows | Enabled, strict, step 25 |
| Screening | Drop short tracks with fewer than 4 observations |
| Lane-change sampling | Require 5 lane changes; keep 1 in 3 negative scenes |
| Maps | Full map |

## Dataset compatibility

Expected raw data layout for this loader.

| Field | Value |
| ----- | ----- |
| Expected release/layout | AD4CHE v1.0 |
| Loader expectation | The loader expects the `AD4CHE_Data_V1.0` directory and matching archive layout. |

## Normalization

### Agent categories

| Dataset type | Dronalize type |
| ------------ | -------------- |
| `car` | `CAR` |
| `truck` | `TRUCK` |
| `bus` | `BUS` |

### Map types

| Dataset type | Dronalize type |
| ------------ | -------------- |
| Extracted lane-image border contour | `LINE_THICK_DASHED` |

## Split support

Use the CLI for current native split and assignment support.

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
