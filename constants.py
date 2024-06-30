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

non_cpk_model_names = ["fashion"]

default_model_name = "vox-adv"

checkpoint_folder_link = "https://github.com/graphemecluster/first-order-model-demo/releases/download/checkpoints"

checkpoint_suffix = ".pth.tar"
config_suffix = "256.yaml"

config_folder = "config"
checkpoint_folder = "checkpoints"

partswap_model_config_dict = {
    "vox-5segments": "vox-256-sem-5segments.yaml",
    "vox-10segments": "vox-256-sem-10segments.yaml",
    "vox-15segments": "vox-256-sem-15segments.yaml",
}

partswap_model_names = list(partswap_model_config_dict.keys())

default_partswap_model_name = "vox-10segments"

seg10_segment_names = [
    "m_0_background",
    "m_1_lf_ear",
    "m_2_mouth",
    "m_3_lf_fr_head",
    "m_4_rt_ear",
    "m_5_frnt_face",
    "m_6_lt_cheek",
    "m_7_eyes",
    "m_8_rt_fr_head",
    "m_9_shoulder",
    "m_10_hair_top",
]

seg_model_dict = {
    "vox-5segments": [],
    "vox-10segments": seg10_segment_names,
    "vox-15segments": [],
}
