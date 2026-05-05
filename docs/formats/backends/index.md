# Storage backends

<div class="section-intro" markdown="1">
`dronalize` supports multiple storage backends so you can choose between simple local persistence,
streaming-oriented sharding, or no-op validation runs.
</div>

## Selecting a backend

The storage backend is a run-level decision set with `--storage-backend` on the
CLI. It cannot at the moment be specified in the TOML config file.

The current default is `pickle` because it requires no extra dependencies.

| Backend | Purpose |
| --- | --- |
| `mds` | Production runs. Writes binary shards and a shard index readable by the Mosaic Streaming library. |
| `pickle` | Simpler alternative to MDS. Writes one pickled [`SceneRecord`](../../reference/api/io/storage-and-manifests.md#dronalize.io.SceneRecord) per scene. No extra dependency. |
| `null` | Debugging and config validation. Runs the full pipeline without writing any files to disk. |

The `null` backend is useful when you want to verify that a config is valid, check how many scenes
a dataset produces, or benchmark the pipeline itself without the cost of I/O.

## Backend pages

- [MDS](mds.md)
- [Pickle](pickle.md)
- [Null](null.md)
