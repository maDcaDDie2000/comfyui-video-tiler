"""
Tile Loop - sequential tile processing.
Chain: Tile Loop Open -> your processing -> Tile Loop Close (tile input).
Requires ComfyUI with comfy_execution and loop infrastructure.
"""

import torch

from .tile_config import parse_tile_config

try:
    from comfy_execution.graph_utils import GraphBuilder
    _HAS_EXECUTION = True
except ImportError:
    _HAS_EXECUTION = False
    GraphBuilder = None


class TileLoopOpen:
    """
    Open a tile loop - outputs current tile and index for each iteration.
    Chain: tile -> your processing -> Tile Loop Close (tile input).
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
    CATEGORY = "Video Tiler"

    def open(self, images, tile_config):
        if not _HAS_EXECUTION or GraphBuilder is None:
            raise RuntimeError(
                "Tile Loop requires ComfyUI with comfy_execution. "
                "Use the parallel workflow: Slice → Get Tile (index 0,1,2...) → your processing → Merge."
            )
        try:
            _, _, _, _, tile_specs = parse_tile_config(tile_config)
            count = len(tile_specs)
            print(f"[Video Tiler] TileLoopOpen: {count} tiles to process")
            graph = GraphBuilder()
            loop_open = graph.node("ForLoopOpen", remaining=count, initial_value1=None)
            remaining = loop_open.out(1)
            sub = graph.node("IntMathOperation", a=count, b=remaining, operation="subtract")
            get_tile = graph.node("GetTile", images=images, tile_config=tile_config, tile_index=sub.out(0))
            return {
                "result": (get_tile.out(0), sub.out(0), tile_config, loop_open.out(0), loop_open.out(2)),
                "expand": graph.finalize(),
            }
        except Exception as e:
            raise RuntimeError(
                "Tile Loop requires loop infrastructure. Use the parallel workflow: "
                "Slice → Get Tile (index 0,1,2...) → your processing → Merge."
            ) from e


class TileLoopClose:
    """
    Close tile loop - receives processed tile directly (no separate Collect Tile).
    Connect: your processing -> tile. Connect Tile Loop Open's accumulation -> accumulation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow_control": ("FLOW_CONTROL", {"rawLink": True}),
                "tile": ("IMAGE", {"forceInput": True}),
                "accumulation": ("ACCUMULATION", {"rawLink": True}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("tiles",)
    FUNCTION = "close"
    CATEGORY = "Video Tiler"

    def close(self, flow_control, tile, accumulation):
        if not _HAS_EXECUTION or GraphBuilder is None:
            raise RuntimeError(
                "Tile Loop requires ComfyUI with comfy_execution. "
                "Use the parallel workflow: Slice → Get Tile → your processing → Merge."
            )
        try:
            graph = GraphBuilder()
            acc_node = graph.node("AccumulateTile", to_add=tile, accumulation=accumulation)
            loop_close = graph.node("ForLoopClose", flow_control=flow_control, initial_value1=acc_node.out(0))
            tiles_list = loop_close.out(0)
            return {
                "result": (tiles_list,),
                "expand": graph.finalize(),
            }
        except Exception as e:
            raise RuntimeError(
                "Tile Loop requires loop infrastructure (internal nodes). "
                "Use the parallel workflow: Slice → Get Tile (index 0,1,2...) → your processing → Merge."
            ) from e
