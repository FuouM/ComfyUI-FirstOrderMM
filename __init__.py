from .run import FOMM_Runner

NODE_CLASS_MAPPINGS = {
    "FOMM_Runner": FOMM_Runner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FOMM_Runner": "FOMM Runner",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']