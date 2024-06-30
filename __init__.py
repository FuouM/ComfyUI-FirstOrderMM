from .run import FOMM_Runner, FOMM_Partswap, FOMM_Seg10Chooser

NODE_CLASS_MAPPINGS = {
    "FOMM_Runner": FOMM_Runner,
    "FOMM_Partswap": FOMM_Partswap,
    "FOMM_Seg10Chooser": FOMM_Seg10Chooser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FOMM_Runner": "FOMM Runner",
    "FOMM_Partswap": "FOMM Partswap",
    "FOMM_Seg10Chooser": "FOMM Seg10 Chooser",
}


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']