supported_model_names = [
    "vox-adv",
    "vox",
    # "taichi",
    # "taichi-adv",
    # "nemo",
    # "mgif",
    # "fashion",
    # "bair",
]

non_cpk_model_names = [
    "fashion"
]

default_model_name = "vox-adv"

checkpoint_folder_link = "https://github.com/graphemecluster/first-order-model-demo/releases/download/checkpoints"

checkpoint_suffix = ".pth.tar"

config_folder = "config"
checkpoint_folder = "checkpoints"

partswap_model_config_dict = {
    "vox-5segments" : "vox-256-sem-5segments.yaml",
    "vox-10segments": "vox-256-sem-10segments.yaml",
    "vox-15segments": "vox-256-sem-15segments.yaml",
}

default_partswap_model_name = "vox-10segments"