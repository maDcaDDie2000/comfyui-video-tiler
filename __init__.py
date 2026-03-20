from .slice_node import VideoTileSlice
from .merge_node import VideoTileMerge
from .get_tile_node import GetTile
from .tile_loop_node import TileLoopOpen, TileLoopClose, TileForLoopOpen, GetTileFromRemaining
from .loop_nodes import (
    AccumulateTile,
    ListLength,
    IntMathOperation,
    IntConditions,
    ForLoopOpen,
    ForLoopClose,
    WhileLoopOpen,
    WhileLoopClose,
)

# Tile Loop requires ForLoop (which uses WhileLoop) + AccumulateTile + IntMath + IntConditions
NODE_CLASS_MAPPINGS = {
    "VideoTileSlice": VideoTileSlice,
    "VideoTileMerge": VideoTileMerge,
    "GetTile": GetTile,
    "TileForLoopOpen": TileForLoopOpen,
    "TileLoopOpen": TileLoopOpen,
    "TileLoopClose": TileLoopClose,
    "GetTileFromRemaining": GetTileFromRemaining,
    "AccumulateTile": AccumulateTile,
    "ListLength": ListLength,
    "IntMathOperation": IntMathOperation,
    "IntConditions": IntConditions,
}
if ForLoopOpen is not None:
    NODE_CLASS_MAPPINGS.update({
        "ForLoopOpen": ForLoopOpen,
        "ForLoopClose": ForLoopClose,
        "WhileLoopOpen": WhileLoopOpen,
        "WhileLoopClose": WhileLoopClose,
    })
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoTileSlice": "Video Tile Slice",
    "VideoTileMerge": "Video Tile Merge",
    "GetTile": "Get Tile",
    "TileForLoopOpen": "Tile For Loop Open",
    "TileLoopOpen": "Tile Loop Open",
    "TileLoopClose": "Tile Loop Close",
    "ForLoopOpen": "For Loop Open",
    "ForLoopClose": "For Loop Close",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
