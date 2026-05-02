from .slice_node import VideoTileSlice
from .fixed_slice_node import VideoTileSliceFixed
from .merge_node import VideoTileMerge
from .soft_overlap_merge_node import VideoTileMergeOverlapSoft
from .get_tile_node import GetTile
from .ref_tile_node import ReferenceTileSlice
from .reference_color_match_node import VideoTileReferenceColorMatch

# Sequential Batcher pattern: Slice outputs tiles as list (OUTPUT_IS_LIST), ComfyUI runs
# downstream once per tile, Merge collects with INPUT_IS_LIST. No custom loop nodes.
NODE_CLASS_MAPPINGS = {
    "VideoTileSlice": VideoTileSlice,
    "VideoTileSliceFixed": VideoTileSliceFixed,
    "VideoTileMerge": VideoTileMerge,
    "VideoTileMergeOverlapSoft": VideoTileMergeOverlapSoft,
    "GetTile": GetTile,
    "ReferenceTileSlice": ReferenceTileSlice,
    "VideoTileReferenceColorMatch": VideoTileReferenceColorMatch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoTileSlice": "Video Tile Slicer (var. size)",
    "VideoTileSliceFixed": "Video Tile Slicer (fixed size)",
    "VideoTileMerge": "Video Tile Merge",
    "VideoTileMergeOverlapSoft": "Video Tile Merge (overlap soft)",
    "GetTile": "Get Tile",
    "ReferenceTileSlice": "Reference Tile Slice",
    "VideoTileReferenceColorMatch": "Video Tile Reference Color Match",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
