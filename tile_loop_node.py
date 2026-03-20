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
            "optional": {
                "remaining_from_loop": ("INT", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "TILE_CONFIG", "FLOW_CONTROL", "ACCUMULATION")
    RETURN_NAMES = ("tile", "tile_index", "tile_config", "flow_control", "accumulation")
    FUNCTION = "open"
    CATEGORY = "Video Tiler"

    def open(self, images, tile_config, remaining_from_loop=None):
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
            # remaining_from_loop = tile_count - len(accumulation) from previous iteration
            remaining = remaining_from_loop if remaining_from_loop is not None else count
            while_open = graph.node(
                "WhileLoopOpen",
                condition=True,
                initial_value0=remaining,
                initial_value1=None,
            )
            sub = graph.node("IntMathOperation", a=count, b=remaining, operation="subtract")
            get_tile = graph.node("GetTile", images=images, tile_config=tile_config, tile_index=sub.out(0))
            return {
                "result": (get_tile.out(0), sub.out(0), tile_config, while_open.out(0), while_open.out(2)),
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
    Connect: your processing -> tile; Tile Loop Open's accumulation -> accumulation;
    Video Tile Slice's tile_count -> tile_count.
    Connect remaining_next -> Tile Loop Open's remaining_from_loop to close the loop.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow_control": ("FLOW_CONTROL", {"rawLink": True}),
                "tile": ("IMAGE", {"forceInput": True}),
                "accumulation": ("ACCUMULATION", {"rawLink": True}),
                "tile_count": ("INT", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("*", "INT")
    RETURN_NAMES = ("tiles", "remaining_next")
    FUNCTION = "close"
    CATEGORY = "Video Tiler"

    def close(self, flow_control, tile, accumulation, tile_count):
        if not _HAS_EXECUTION or GraphBuilder is None:
            raise RuntimeError(
                "Tile Loop requires ComfyUI with comfy_execution. "
                "Use the parallel workflow: Slice → Get Tile → your processing → Merge."
            )
        try:
            graph = GraphBuilder()
            acc_node = graph.node("AccumulateTile", to_add=tile, accumulation=accumulation)
            len_node = graph.node("ListLength", lst=acc_node.out(0))
            # cond: tile_count > len(accumulation) — no link to open node needed
            cond = graph.node("IntConditions", a=tile_count, b=len_node.out(0), operation=">")
            # remaining_next = tile_count - len(accumulation) for next iteration
            remaining_next = graph.node("IntMathOperation", operation="subtract", a=tile_count, b=len_node.out(0))
            loop_close = graph.node(
                "WhileLoopClose",
                flow_control=flow_control,
                condition=cond.out(0),
                initial_value0=remaining_next.out(0),
                initial_value1=acc_node.out(0),
            )
            tiles_list = loop_close.out(1)
            return {
                "result": (tiles_list, remaining_next.out(0)),
                "expand": graph.finalize(),
            }
        except Exception as e:
            raise RuntimeError(
                "Tile Loop requires loop infrastructure (internal nodes). "
                "Use the parallel workflow: Slice → Get Tile (index 0,1,2...) → your processing → Merge."
            ) from e
