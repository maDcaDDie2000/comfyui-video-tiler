"""
Tile Loop - self-contained sequential tile processing with user's chain.
TileLoopOpen + TileLoopClose: process tiles one-by-one, chain your nodes in between.
Requires ComfyUI with comfy_execution (built-in).
"""

import torch

from .tile_config import parse_tile_config
from .loop_nodes import _HAS_EXECUTION

if _HAS_EXECUTION:
    from comfy_execution.graph_utils import GraphBuilder


class TileLoopOpen:
    """
    Open a tile loop - outputs current tile and index for each iteration.
    Chain your processing: tile -> your nodes -> Accumulate -> TileLoopClose.
    Fully self-contained, no external node packs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "tile_config": ("TILE_CONFIG", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "TILE_CONFIG", "FLOW_CONTROL", "ACCUMULATION")
    RETURN_NAMES = ("tile", "tile_index", "tile_config", "flow_control", "accumulation")
    FUNCTION = "open"
    CATEGORY = "video"

    def open(self, images, tile_config):
        if not _HAS_EXECUTION:
            raise RuntimeError(
                "Tile Loop requires ComfyUI with comfy_execution. "
                "Update ComfyUI or use Video Tile Process Loop (passthrough) instead."
            )
        _, _, _, _, tile_specs = parse_tile_config(tile_config)
        count = len(tile_specs)

        graph = GraphBuilder()
        loop_open = graph.node("ForLoopOpen", remaining=count, initial_value1=count)
        remaining = loop_open.out(1)
        sub = graph.node("IntMathOperation", a=count, b=remaining, operation="subtract")
        get_tile = graph.node("GetTile", images=images, tile_config=tile_config, tile_index=sub.out(0))

        return {
            "result": (get_tile.out(0), sub.out(0), tile_config, loop_open.out(0), loop_open.out(2)),
            "expand": graph.finalize(),
        }


class TileLoopClose:
    """
    Close tile loop - collects processed tiles from each iteration.
    Connect: your processing -> Accumulate -> TileLoopClose (initial_value1=accumulation).
    Output: list of tiles for Merge.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow_control": ("FLOW_CONTROL", {"rawLink": True}),
            },
            "optional": {
                "initial_value1": ("ACCUMULATION", {"rawLink": True}),
                "initial_value2": ("*", {"rawLink": True}),
                "initial_value3": ("*", {"rawLink": True}),
                "initial_value4": ("*", {"rawLink": True}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("tiles",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "close"
    CATEGORY = "video"

    def close(self, flow_control, **kwargs):
        if not _HAS_EXECUTION:
            raise RuntimeError("Tile Loop requires ComfyUI with comfy_execution.")
        graph = GraphBuilder()
        loop_close = graph.node("ForLoopClose", flow_control=flow_control, **kwargs)
        acc_to_list = graph.node("AccumulationToListNode", accumulation=loop_close.out(0))
        return {
            "result": (acc_to_list.out(0),),
            "expand": graph.finalize(),
        }
