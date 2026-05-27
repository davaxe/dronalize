# Storage backends

<div class="section-intro" markdown="1">
`dronalize` supports multiple storage backends so you can choose between simple local persistence,
streaming-oriented sharding, or no-op validation runs.
</div>

## Selecting a backend

The storage backend is a run-level decision set with `--storage-backend` on the
CLI. It cannot at the moment be specified in the TOML config file.

The current default is `pickle` because it requires no extra dependencies. Built-in backends are
registered by name, and Python integrations may register additional backend names before resolving a
request.

| Backend | Purpose |
| --- | --- |
| `mds` | Production runs. Writes binary shards and a shard index readable by the Mosaic Streaming library. |
| `pickle` | Simpler alternative to MDS. Writes one pickled [`SceneRecord`](../../reference/api/io/storage-and-manifests.md#dronalize.io.SceneRecord) per scene. No extra dependency. |
| `null` | Debugging and config validation. Runs the full pipeline without writing any files to disk. |

The `null` backend is useful when you want to verify that a config is valid, check how many scenes
a dataset produces, or benchmark the pipeline itself without the cost of I/O.

## Custom backends

Custom writers can be registered through
[`register_writer_backend()`](../../reference/api/io/backends.md#dronalize.io.backends.register_writer_backend).
The registry key is a string, so application code can pass that same string as
`ExecutionRequest.storage_backend` after registration.

<!-- no-validate -->
```python
from dronalize.io.backends import register_writer_backend

register_writer_backend("my-backend", build_my_writer_factory)
```

CLI selection is still just `--storage-backend <name>`, but the backend must already be registered in
the Python process that resolves the request.

## Custom samples

The built-in `pickle` and `mds` writers also support Python-level sample
customization without registering a new storage backend. Pass `output_sample`
to an `ExecutionRequest` to have the writer call that function on each sample before writing.

Custom MDS samples must also provide explicit `mds_columns`, because the MDS
writer needs the column schema before the first sample is written.

## Backend pages

- [MDS](mds.md)
- [Pickle](pickle.md)
- [Null](null.md)
