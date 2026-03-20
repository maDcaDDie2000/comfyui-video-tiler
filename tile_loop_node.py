"""
Tile Loop - sequential tile processing.
Matches execution-inversion-demo: user adds For Loop Open so flow_control points to it.
Chain: For Loop Open -> Tile Loop Open (remaining) -> processing -> Tile Loop Close.
"""

from .tile_config import parse_tile_config
from .get_tile_node import get_tile_by_index

try:
    from comfy_execution.graph_utils import GraphBuilder
    _HAS_EXECUTION = True
except ImportError:
    _HAS_EXECUTION = False
    GraphBuilder = None


class TileLoopOpen:
    """
    Get current tile from loop. Connect For Loop Open's remaining -> remaining.
    Chain: For Loop Open -> Tile Loop Open -> your processing -> Tile Loop Close.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "tile_config": ("TILE_CONFIG", {"forceInput": True}),
                "remaining": ("INT", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "TILE_CONFIG")
    RETURN_NAMES = ("tile", "tile_index", "tile_config")
    FUNCTION = "open"
    CATEGORY = "Video Tiler"

    def open(self, images, tile_config, remaining):
        if not _HAS_EXECUTION or GraphBuilder is None:
            raise RuntimeError(
                "Tile Loop requires ComfyUI with comfy_execution. "
                "Use the parallel workflow: Slice → Get Tile (index 0,1,2...) → your processing → Merge."
            )
        try:
            _, _, _, _, tile_specs = parse_tile_config(tile_config)
            count = len(tile_specs)
            tile_index = max(0, min(count - remaining, count - 1))
            tile = get_tile_by_index(images, tile_config, tile_index)
            return (tile, tile_index, tile_config)
        except Exception as e:
            raise RuntimeError(
                "Tile Loop requires loop infrastructure. Use the parallel workflow: "
                "Slice → Get Tile (index 0,1,2...) → your processing → Merge."
            ) from e


class TileLoopClose:
    """
    Close tile loop - wraps For Loop Close + AccumulateTile.
    Connect: For Loop Open's flow_control -> flow_control; your processing -> tile;
    For Loop Open's value1 -> accumulation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow_control": ("FLOW_CONTROL", {"rawLink": True}),
                "tile": ("IMAGE", {"forceInput": True}),
                "accumulation": ("*", {"rawLink": True}),
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
