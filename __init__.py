from .slice_node import VideoTileSlice
from .merge_node import VideoTileMerge
from .get_tile_node import GetTile

# Sequential Batcher pattern: Slice outputs tiles as list (OUTPUT_IS_LIST), ComfyUI runs
# downstream once per tile, Merge collects with INPUT_IS_LIST. No custom loop nodes.
NODE_CLASS_MAPPINGS = {
    "VideoTileSlice": VideoTileSlice,
    "VideoTileMerge": VideoTileMerge,
    "GetTile": GetTile,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoTileSlice": "Video Tile Slice",
    "VideoTileMerge": "Video Tile Merge",
    "GetTile": "Get Tile",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
