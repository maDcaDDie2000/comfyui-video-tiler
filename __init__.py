from .slice_node import VideoTileSlice
from .merge_node import VideoTileMerge
from .get_tile_node import GetTile, GetTileCount
from .tile_loop_node import RemainingToIndex, TileLoopOpen, TileLoopClose
from .process_loop_node import VideoTileProcessLoop
from .loop_nodes import (
    IntMathOperation,
    IntConditions,
    AccumulateNode,
    AccumulationToListNode,
    _HAS_EXECUTION,
)

# Build mappings - loop nodes only when comfy_execution available
_node_mappings = {
    "VideoTileSlice": VideoTileSlice,
    "VideoTileMerge": VideoTileMerge,
    "GetTile": GetTile,
    "GetTileCount": GetTileCount,
    "RemainingToIndex": RemainingToIndex,
    "VideoTileProcessLoop": VideoTileProcessLoop,
    "AccumulateNode": AccumulateNode,
    "AccumulationToListNode": AccumulationToListNode,
    "IntMathOperation": IntMathOperation,
    "IntConditions": IntConditions,
}

_display_mappings = {
    "VideoTileSlice": "Video Tile Slice",
    "VideoTileMerge": "Video Tile Merge",
    "GetTile": "Get Tile",
    "GetTileCount": "Get Tile Count",
    "RemainingToIndex": "Remaining to Index",
    "VideoTileProcessLoop": "Video Tile Process Loop",
    "AccumulateNode": "Accumulate",
    "AccumulationToListNode": "Accumulation to List",
    "IntMathOperation": "Int Math",
    "IntConditions": "Int Condition",
}

if _HAS_EXECUTION:
    from .loop_nodes import WhileLoopOpen, WhileLoopClose, ForLoopOpen, ForLoopClose
    _node_mappings.update({
        "TileLoopOpen": TileLoopOpen,
        "TileLoopClose": TileLoopClose,
        "WhileLoopOpen": WhileLoopOpen,
        "WhileLoopClose": WhileLoopClose,
        "ForLoopOpen": ForLoopOpen,
        "ForLoopClose": ForLoopClose,
    })
    _display_mappings.update({
        "TileLoopOpen": "Tile Loop Open",
        "TileLoopClose": "Tile Loop Close",
        "WhileLoopOpen": "While Loop Open",
        "WhileLoopClose": "While Loop Close",
        "ForLoopOpen": "For Loop Open",
        "ForLoopClose": "For Loop Close",
    })

NODE_CLASS_MAPPINGS = _node_mappings
NODE_DISPLAY_NAME_MAPPINGS = _display_mappings

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
