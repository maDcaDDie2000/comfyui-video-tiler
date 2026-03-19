from .slice_node import VideoTileSlice
from .merge_node import VideoTileMerge
from .get_tile_node import GetTile
from .tile_loop_node import TileLoopOpen, TileLoopClose
from .loop_nodes import (
    IntMathOperation,
    IntConditions,
    AccumulateNode,
    AccumulationToListNode,
    _HAS_EXECUTION,
)

_node_mappings = {
    "VideoTileSlice": VideoTileSlice,
    "VideoTileMerge": VideoTileMerge,
    "GetTile": GetTile,
    "AccumulateNode": AccumulateNode,
}

_display_mappings = {
    "VideoTileSlice": "Video Tile Slice",
    "VideoTileMerge": "Video Tile Merge",
    "GetTile": "Get Tile",
    "AccumulateNode": "Collect Tile",
}

_internal_mappings = {
    "IntMathOperation": IntMathOperation,
    "IntConditions": IntConditions,
    "AccumulationToListNode": AccumulationToListNode,
}

_internal_display = {
    "IntMathOperation": "Int Math",
    "IntConditions": "Int Condition",
    "AccumulationToListNode": "Accumulation to List",
}

if _HAS_EXECUTION:
    from .loop_nodes import WhileLoopOpen, WhileLoopClose, ForLoopOpen, ForLoopClose
    _node_mappings.update({
        "TileLoopOpen": TileLoopOpen,
        "TileLoopClose": TileLoopClose,
    })
    _display_mappings.update({
        "TileLoopOpen": "Tile Loop Open",
        "TileLoopClose": "Tile Loop Close",
    })
    _internal_mappings.update({
        "WhileLoopOpen": WhileLoopOpen,
        "WhileLoopClose": WhileLoopClose,
        "ForLoopOpen": ForLoopOpen,
        "ForLoopClose": ForLoopClose,
    })
    _internal_display.update({
        "WhileLoopOpen": "While Loop Open",
        "WhileLoopClose": "While Loop Close",
        "ForLoopOpen": "For Loop Open",
        "ForLoopClose": "For Loop Close",
    })

_node_mappings.update(_internal_mappings)
_display_mappings.update(_internal_display)

NODE_CLASS_MAPPINGS = _node_mappings
NODE_DISPLAY_NAME_MAPPINGS = _display_mappings

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
