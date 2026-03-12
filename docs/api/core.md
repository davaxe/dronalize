# Core API

The core API is split into four focused packages, each answering a single question about the system.

---

## Module Overview

| Package | Description |
|---------|-------------|
| `models` | Domain objects — `Scene`, agent/edge categories, dataset splits |
| `maps` | Map/road geometry — graph data structures, builders, and resolvers |
| `loading` | Data ingestion — source discovery, scene loaders, and writers |
| `exceptions` | Exception hierarchy |

---

### `dronalize.scene`

::: dronalize.scene

---

### `dronalize.categories`

::: dronalize.categories

---

## Maps

Map handling: the `MapGraph` data structure, builder abstractions, and resolver protocols.

### `dronalize.maps`

::: dronalize.maps
    options:
      show_submodules: true

---

### `dronalize.maps.graph`

::: dronalize.maps.graph

---

### `dronalize.maps.builder`

::: dronalize.maps.builder

---

### `dronalize.maps.resolver`

::: dronalize.maps.resolver

---

## Loading

Data loading: source discovery, scene loader protocols and base classes, and scene writers.

### `dronalize.loading`

::: dronalize.loading
    options:
      show_submodules: true

---

### `dronalize.loading.loader`

::: dronalize.loading.loader

---

### `dronalize.loading.writer`

::: dronalize.loading.writer

---

## Exceptions

Top-level exception hierarchy for the library.

### `dronalize.exceptions`

::: dronalize.exceptions
