from .run import (
    FOMM_Partswap,
    FOMM_Runner,
    FOMM_Seg5Chooser,
    FOMM_Seg10Chooser,
    FOMM_Seg15Chooser,
    Articulate_Runner,
    Spline_Runner
)

NODE_CLASS_MAPPINGS = {
    "FOMM_Runner": FOMM_Runner,
    "FOMM_Partswap": FOMM_Partswap,
    "FOMM_Seg5Chooser": FOMM_Seg5Chooser,
    "FOMM_Seg10Chooser": FOMM_Seg10Chooser,
    "FOMM_Seg15Chooser": FOMM_Seg15Chooser,
    "Articulate_Runner": Articulate_Runner,
    "Spline_Runner": Spline_Runner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FOMM_Runner": "FOMM Runner",
    "FOMM_Partswap": "FOMM Partswap",
    "FOMM_Seg5Chooser": "FOMM Seg5 Chooser",
    "FOMM_Seg10Chooser": "FOMM Seg10 Chooser",
    "FOMM_Seg15Chooser": "FOMM Seg15 Chooser",
    "Articulate_Runner": "Articulate Runner",
    "Spline_Runner": "Spline Runner",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
