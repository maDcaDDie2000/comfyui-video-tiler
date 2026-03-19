from .slice_node import VideoTileSlice
from .merge_node import VideoTileMerge
from .get_tile_node import GetTile
from .tile_loop_node import TileLoopOpen, TileLoopClose
from .loop_nodes import (
    AccumulateTile,
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
    "TileLoopOpen": TileLoopOpen,
    "TileLoopClose": TileLoopClose,
    "AccumulateTile": AccumulateTile,
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
    "TileLoopOpen": "Tile Loop Open",
    "TileLoopClose": "Tile Loop Close",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
