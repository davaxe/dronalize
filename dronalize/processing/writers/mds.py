import multiprocessing as mp
from pathlib import Path

import numpy as np
from streaming import MDSWriter

from dronalize.core.datatypes.scene import Scene
from dronalize.core.datatypes.split import DatasetSplit
from dronalize.core.protocols.writer import BaseSceneWriter

# -- Column schema ----------------------------------------------------------
# Every field is stored as a native MDS type: int scalars, or ndarray with a
# fixed dtype (but dynamic shape, since agent count N varies per scene).
#
# The trajectory arrays have shape (N, T, D) where T is input_len or
# output_len and D is the feature dimension (2 for pos/vel/acc, 1 for yaw).
# Masks have shape (N, T).  MDS does not support bool arrays, so masks are
# stored as uint8 (0/1).
#
# Map arrays have shape (num_map_nodes, 2), (2, num_map_edges), etc.
# When no map is available, 1-element placeholder arrays are written
# (MDS rejects 0-element arrays) and ``has_map`` is set to 0.

MDS_COLUMNS = {
    # -- scalar metadata --
    "scene_number": "int",
    "target_agent_id": "int",
    "num_nodes": "int",
    "ta_index": "int",
    "input_len": "int",
    "output_len": "int",
    # -- agent arrays (dynamic shape, fixed dtype) --
    "type": "ndarray:int32",
    "inp_pos": "ndarray:float32",
    "inp_vel": "ndarray:float32",
    "inp_acc": "ndarray:float32",
    "inp_yaw": "ndarray:float32",
    "trg_pos": "ndarray:float32",
    "trg_vel": "ndarray:float32",
    "trg_acc": "ndarray:float32",
    "trg_yaw": "ndarray:float32",
    # MDS does not support ndarray:bool — store masks as uint8 (0/1)
    "input_mask": "ndarray:uint8",
    "valid_mask": "ndarray:uint8",
    "ma_mask": "ndarray:uint8",
    "sa_mask": "ndarray:uint8",
    # -- map arrays (dynamic shape, fixed dtype) --
    "has_map": "int",
    "map_num_nodes": "int",
    "map_num_edges": "int",
    "map_node_positions": "ndarray:float32",
    "map_edge_indices": "ndarray:int64",
    "map_node_types": "ndarray:int64",
    "map_edge_types": "ndarray:int64",
}

# 1-element placeholder arrays for map-less samples.
# MDS rejects 0-element arrays, so we use minimal placeholders
# and signal absence via ``has_map == 0``.
_PLACEHOLDER_F32 = np.zeros((1,), dtype=np.float32)
_PLACEHOLDER_I64 = np.zeros((1,), dtype=np.int64)


class MDSSceneWriter(BaseSceneWriter):
    """Write (scene, target_agent) samples to MDS format.

    Each scene is expanded into one or more samples via
    ``Scene.to_numpy_dict``.  The map graph (if available) is serialized
    alongside each sample so that every MDS row is fully self-contained.

    Parameters
    ----------
    output_dir : str
        Root output directory for MDS shards.
    parallel : bool
        If True, appends the process name to the output path to avoid
        write conflicts across workers.
    multiple_targets : bool or int
        Forwarded to ``Scene.to_numpy_dict``.  Controls how many target-
        agent samples are produced per scene.
    compression : str or None
        MDS compression algorithm, e.g. ``"zstd"`` or ``"zstd:7"``.
    size_limit : str or int
        Maximum uncompressed shard size (default ``"67mb"``).

    """

    def __init__(
        self,
        output_dir: str,
        *,
        parallel: bool = False,
        multiple_targets: bool | int = False,
        compression: str | None = None,
        size_limit: str | int = "67mb",
    ) -> None:
        super().__init__(parallel=parallel)
        self._base_output_dir: Path = Path(output_dir)
        self._multiple_targets = multiple_targets

        if parallel:
            self._base_output_dir /= mp.current_process().name

        self._inner: MDSWriter = MDSWriter(
            out=str(self._base_output_dir),
            columns=MDS_COLUMNS,
            compression=compression,
            size_limit=size_limit,
        )

    def write(self, processed: Scene) -> None:
        # Resolve the map once per scene (shared across target-agent samples)
        map_sample = self._encode_map(processed)

        samples = processed.to_numpy_dict(multiple_targets=self._multiple_targets)
        for target_id, numpy_dict in samples.items():
            sample = {
                # -- scalar metadata --
                "scene_number": processed.scene_number,
                "target_agent_id": int(target_id),
                "num_nodes": int(numpy_dict["num_nodes"]),
                "ta_index": int(numpy_dict["ta_index"]),
                "input_len": processed.input_len,
                "output_len": processed.output_len,
                # -- agent arrays --
                "type": numpy_dict["type"],
                "inp_pos": numpy_dict["inp_pos"],
                "inp_vel": numpy_dict["inp_vel"],
                "inp_acc": numpy_dict["inp_acc"],
                "inp_yaw": numpy_dict["inp_yaw"],
                "trg_pos": numpy_dict["trg_pos"],
                "trg_vel": numpy_dict["trg_vel"],
                "trg_acc": numpy_dict["trg_acc"],
                "trg_yaw": numpy_dict["trg_yaw"],
                # Cast bool → uint8 (MDS does not support bool arrays)
                "input_mask": numpy_dict["input_mask"].astype(np.uint8),
                "valid_mask": numpy_dict["valid_mask"].astype(np.uint8),
                "ma_mask": numpy_dict["ma_mask"].astype(np.uint8),
                "sa_mask": numpy_dict["sa_mask"].astype(np.uint8),
                # -- map --
                **map_sample,
            }
            self._inner.write(sample)

    @staticmethod
    def _encode_map(scene: Scene) -> dict:
        """Encode the scene's map graph into flat MDS-compatible columns."""
        graph = scene.resolve_map()
        if graph is None:
            return {
                "has_map": 0,
                "map_num_nodes": 0,
                "map_num_edges": 0,
                "map_node_positions": _PLACEHOLDER_F32,
                "map_edge_indices": _PLACEHOLDER_I64,
                "map_node_types": _PLACEHOLDER_I64,
                "map_edge_types": _PLACEHOLDER_I64,
            }
        return {
            "has_map": 1,
            "map_num_nodes": graph.num_nodes,
            "map_num_edges": graph.num_edges,
            "map_node_positions": graph.node_positions,  # (N, 2)  float32
            "map_edge_indices": graph.edge_indices,  # (2, M)  int64
            "map_node_types": graph.node_types,  # (N,)    int64
            "map_edge_types": graph.edge_types,  # (M,)    int64
        }

    def finalize(self) -> None:
        self._inner.finish()

    def support_parallel(self) -> bool:
        return True


if __name__ == "__main__":
    from dronalize.datasets.eth_ucy import HotelLoader

    path = Path("data")
    loader = HotelLoader(path, split=DatasetSplit.TRAIN)
    # Example usage
    writer = MDSSceneWriter("output/scenes", parallel=False)
    count = 0
    for scene in loader.scenes():
        count += 1
        writer.write(scene)
    writer.finalize()

    print(f"Wrote {count} samples to MDS format in 'output/scenes'")

    # Tes load
    from streaming import StreamingDataset

    dataset = StreamingDataset(local="output/scenes", batch_size=1)
    print(f"Dataset has {dataset.num_samples} samples")
    # for i in range(dataset.num_samples):
    #     batch = dataset[i]
    #     print(batch)
    #     break
