# Datasets

Dronalize supports a wide range of trajectory prediction datasets, spanning drone-captured bird's-eye-view recordings, autonomous vehicle sensor suites, and pedestrian trajectory benchmarks.

This section provides dataset-specific documentation including download instructions, preprocessing commands, and dataset overviews.

---

## Dataset Categories

### Drone Datasets

Bird's-eye-view trajectory datasets captured from drones or elevated camera positions. These include highway, roundabout, intersection, and urban scenarios.

→ [Drone Datasets](drone.md)

### AV Datasets

Large-scale motion forecasting benchmarks collected from instrumented vehicles, typically including HD maps, LiDAR, and multi-sensor data.

→ [AV Datasets](av.md)

### Pedestrian Datasets

Pedestrian trajectory datasets commonly used for social force and crowd behavior modeling.

→ [Pedestrian Datasets](pedestrian.md)

---

## Supported Datasets at a Glance

| Dataset | Category | Sampling Rate | Map Data |
|---------|----------|--------------|----------|
| highD | Drone | 25 Hz | ✗ |
| rounD | Drone | 25 Hz | ✗ |
| inD | Drone | 25 Hz | ✗ |
| exiD | Drone | 25 Hz | ✗ |
| uniD | Drone | 25 Hz | ✗ |
| openDD | Drone | 10 Hz | ✗ |
| SIND | Drone | 30 Hz | ✗ |
| A43 | Drone | 25 Hz | ✗ |
| AD4CHE | Drone | 25 Hz | ✗ |
| INTERACTION | Drone | 10 Hz | ✓ |
| Argoverse 1 | AV | 10 Hz | ✓ |
| Argoverse 2 | AV | 10 Hz | ✓ |
| Waymo (WOMD) | AV | 10 Hz | ✓ |
| nuScenes | AV | 2 Hz | ✓ |
| Lyft Level 5 | AV | 10 Hz | ✓ |
| View-of-Delft | AV | 10 Hz | ✓ |
| ApolloScape | AV | 2 Hz | ✗ |
| I-80 (NGSIM) | AV | 10 Hz | ✗ |
| US-101 (NGSIM) | AV | 10 Hz | ✗ |
| ETH | Pedestrian | 2.5 Hz | ✗ |
| UCY | Pedestrian | 2.5 Hz | ✗ |

!!! note
    Sampling rates listed above are the native recording rates. Dronalize supports resampling to different target rates (e.g., 5 Hz or 10 Hz) during preprocessing via configuration files.