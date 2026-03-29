from .slice_node import VideoTileSlice
from .fixed_slice_node import VideoTileSliceFixed
from .merge_node import VideoTileMerge
from .get_tile_node import GetTile
from .ref_tile_node import ReferenceTileSlice

# Sequential Batcher pattern: Slice outputs tiles as list (OUTPUT_IS_LIST), ComfyUI runs
# downstream once per tile, Merge collects with INPUT_IS_LIST. No custom loop nodes.
NODE_CLASS_MAPPINGS = {
    "VideoTileSlice": VideoTileSlice,
    "VideoTileSliceFixed": VideoTileSliceFixed,
    "VideoTileMerge": VideoTileMerge,
    "GetTile": GetTile,
    "ReferenceTileSlice": ReferenceTileSlice,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoTileSlice": "Variable Tile Size",
    "VideoTileSliceFixed": "Fixed Tile Size",
    "VideoTileMerge": "Video Tile Merge",
    "GetTile": "Get Tile",
    "ReferenceTileSlice": "Reference Tile Slice",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
