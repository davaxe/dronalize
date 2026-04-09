# Hotel

<div class="section-intro" markdown="1">
Hotel is a pedestrian-only scene from the ETH/UCY trajectory benchmark family. It is typically used for socially aware human motion forecasting in open public spaces with interacting walkers.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Pedestrian</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Pedestrians</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Camera</strong></div>
  <div class="summary-item"><span>Map context</span><strong>None</strong></div>
  <div class="summary-item"><span># Samples</span><strong>Processed samples planned</strong></div>
</div>

## Dataset facts

| Field               | Value                               | Notes                                                                    |
| ------------------- | ----------------------------------- | ------------------------------------------------------------------------ |
| Release year        | 2007                                | Part of the UCY scene family used with ETH/UCY benchmarks.               |
| Domain              | Pedestrian                          | Used for interaction-aware human trajectory prediction.                  |
| Capture platform    | Overhead pedestrian scene recording | Focused on walker trajectories in a shared public space.                 |
| Primary agent types | Pedestrians                         | Human motion is the only target class.                                   |
| Map context         | Limited                             | Usually treated as a scene-layout benchmark rather than a map benchmark. |
| Benchmark family    | ETH/UCY                             | Shares setup conventions with `eth`, `univ`, `zara1`, and `zara2`.       |
| Data format         | Text trajectory files               | Commonly arranged into train, validation, and test folders.              |

## Default processing profile

These are the default Dronalize settings used when processing this dataset.

| Setting | Default |
| ------- | ------- |
| Source sequence | 8 obs / 12 pred @ 2.5 Hz |
| Effective sequence | 29 obs / 48 pred @ 10 Hz |
| Resampling | Linear 4:1 |
| Windowing | 20-frame window, step 1 |
| Filtering | Keep agents with at least 2 samples |
| Maps | Disabled |

## Split support

Use the command below for the most up-to-date split support information for this dataset, including native splits, supported custom split strategies, and any recommended strategy.

```bash
dronalize split-support hotel
```

## References

- Family reference: [You'll never walk alone: Modeling social behavior for multi-target tracking](https://ieeexplore.ieee.org/document/5459260)

## Expected structure

```text
ethucy/
└── hotel/
    ├── train/
    ├── val/
    └── test/
```
