from __future__ import annotations

from dronalize.core.graph.builder import GraphBuilder


class A43GraphBuilder(GraphBuilder):
    """Graph builder for the A43 dataset.

    This dataset does not have actual map data, but the map can be reconstructed
    (inferred/estimated) from the trajectories of the vehicles. The graph
    builder uses a mathematical heuristic to infer the lane structure based on
    lane width and the number of lanes.

    """
