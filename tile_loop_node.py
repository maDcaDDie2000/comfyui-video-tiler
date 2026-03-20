"""
Tile Loop - sequential tile processing.
TileForLoopOpen embeds tile extraction inside the loop so the loop feeds the current tile each iteration.
"""

from .tile_config import parse_tile_config
from .get_tile_node import get_tile_by_index

try:
    from comfy_execution.graph_utils import GraphBuilder
    _HAS_EXECUTION = True
except ImportError:
    _HAS_EXECUTION = False
    GraphBuilder = None


class GetTileFromRemaining:
    """
    Internal: get tile by remaining count. remaining=N -> tile index 0, remaining=N-1 -> index 1, etc.
    Used inside TileForLoopOpen expansion so it gets cloned with the loop.
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
    FUNCTION = "get"
    CATEGORY = "Video Tiler"

    def get(self, images, tile_config, remaining):
        if isinstance(remaining, (list, tuple)):
            remaining = remaining[0] if remaining else 0
        _, _, _, _, tile_specs = parse_tile_config(tile_config)
        count = len(tile_specs)
        tile_index = max(0, min(count - remaining, count - 1))
        tile = get_tile_by_index(images, tile_config, tile_index)
        return (tile, tile_index, tile_config)


class TileForLoopOpen:
    """
    Tile loop open - loop feeds current tile each iteration.
    Tile extraction is inside the loop body, so each iteration gets the correct tile.
    Chain: Slice -> Tile For Loop Open -> your processing -> Tile Loop Close.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "tile_config": ("TILE_CONFIG", {"forceInput": True}),
                "tile_count": ("INT", {"default": 1, "min": 1, "max": 100000}),
            },
        }

    RETURN_TYPES = ("FLOW_CONTROL", "INT", "*", "IMAGE", "INT", "TILE_CONFIG")
    RETURN_NAMES = ("flow_control", "remaining", "accumulation", "tile", "tile_index", "tile_config")
    FUNCTION = "open"
    CATEGORY = "Video Tiler"

    def open(self, images, tile_config, tile_count):
        if not _HAS_EXECUTION or GraphBuilder is None:
            raise RuntimeError(
                "Tile Loop requires ComfyUI with comfy_execution. "
                "Use the parallel workflow: Slice → Get Tile (index 0,1,2...) → your processing → Merge."
            )
        try:
            graph = GraphBuilder()
            # WhileLoopOpen: value0=remaining, value1=accumulation (None initially)
            while_open = graph.node(
                "WhileLoopOpen",
                condition=tile_count,
                initial_value0=tile_count,
                initial_value1=None,
            )
            # GetTileFromRemaining is INSIDE the expansion - gets cloned each iteration with new remaining
            get_tile = graph.node(
                "GetTileFromRemaining",
                images=images,
                tile_config=tile_config,
                remaining=[while_open, 1],  # value0 = remaining
            )
            return {
                "result": (
                    "stub",
                    tile_count,
                    None,  # accumulation (value1)
                    get_tile.out(0),
                    get_tile.out(1),
                    get_tile.out(2),
                ),
                "expand": graph.finalize(),
            }
        except Exception as e:
            raise RuntimeError(
                "Tile Loop requires loop infrastructure. Use the parallel workflow: "
                "Slice → Get Tile (index 0,1,2...) → your processing → Merge."
            ) from e


class TileLoopOpen:
    """
    Legacy: Get current tile from loop (remaining input).
    Prefer Tile For Loop Open - it feeds the tile directly each iteration.
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
        if isinstance(remaining, (list, tuple)):
            remaining = remaining[0] if remaining else 0
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
