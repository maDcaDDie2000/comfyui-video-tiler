from .slice_node import VideoTileSlice
from .merge_node import VideoTileMerge
from .get_tile_node import GetTile
from .tile_loop_node import TileLoopOpen, TileLoopClose

# User-facing nodes only - internal nodes removed per user request
# Note: Tile Loop requires loop infrastructure to function; use parallel workflow if it fails
NODE_CLASS_MAPPINGS = {
    "VideoTileSlice": VideoTileSlice,
    "VideoTileMerge": VideoTileMerge,
    "GetTile": GetTile,
    "TileLoopOpen": TileLoopOpen,
    "TileLoopClose": TileLoopClose,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoTileSlice": "Video Tile Slice",
    "VideoTileMerge": "Video Tile Merge",
    "GetTile": "Get Tile",
    "TileLoopOpen": "Tile Loop Open",
    "TileLoopClose": "Tile Loop Close",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
