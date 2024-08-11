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

face_parser_checkpoint_name = "79999_iter.pth"

partswap_model_config_dict = {
    "vox-5segments": "vox-256-sem-5segments.yaml",
    "vox-10segments": "vox-256-sem-10segments.yaml",
    "vox-15segments": "vox-256-sem-15segments.yaml",
    "vox-cpk": "vox-256-sem-10segments.yaml",
    "vox-adv-cpk": "vox-256-sem-10segments.yaml",
    "vox-first-order": "vox-256-sem-10segments.yaml",
}

partswap_face_parser_enforce_models = ["vox-first-order"]

partswap_fomm_model_names = ["vox-cpk", "vox-adv-cpk", "vox-first-order"]

partswap_model_length_dict = {
    "vox-5segments": 6,
    "vox-10segments": 11,
    "vox-15segments": 16,
    "vox-cpk": 11,
    "vox-adv-cpk": 11,
    "vox-first-order": 11,
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

seg5_segment_names = [
    "m_0_background",
    "m_1_chin",
    "m_2_frnt_face",
    "m_3_shoulder",
    "m_4_hair",
    "m_5_fr_head",
]

seg_15_segment_names = [
    "m_0_background",
    "m_1_rt_fr_head",
    "m_2_eyes",
    "m_3_mouth",
    "m_4_lwr_shoulder",
    "m_5_shoulder",
    "m_6_lt_fr_head",
    "m_7",
    "m_8_hair_tip",
    "m_9_frnt_face",
    "m_10_rt_ear",
    "m_11_outer_hair",
    "m_12_lt_ear",
    "m_13_lt_hair",
    "m_14_rt_hair",
    "m_15_neck",
]

seg_model_dict = {
    "vox-5segments": seg5_segment_names,
    "vox-10segments": seg10_segment_names,
    "vox-15segments": seg_15_segment_names,
    "vox": seg10_segment_names,
    "vox-adv": seg10_segment_names,
}

ARTICULATE_MODEL_PATH = "module_articulate/models/vox256.pth"
ARTICULATE_CFG_PATH = "module_articulate/config/vox256.yaml"

SPLINE_MODES = ["relative", "standard", "avd"]
SPLINE_DEFAULT = "relative"

SPLINE_MODEL_PATH = "module_spline/models/vox.pth.tar"
SPLINE_CFG_PATH = "module_spline/config/vox-256.yaml"

MRFA_MODEL_NAMES = ["vox", "celebvhq"]
MRFA_MODEL_PATHS = {
    "vox": "module_mrfa/models/vox.pth",
    "celebvhq": "module_mrfa/models/celebvhq.pth",
}
MRFA_DEFAULT_MODEL = "vox"

MRFA_CFG_PATHS = {
    "celebvhq": "module_mrfa/configs/celebvhq.yaml",
    "vox": "module_mrfa/configs/vox1.yaml",
}

FSRT_MODEL_NAMES = [
    "vox256", "vox256_2Source"
]

FSRT_DEFAULT_MODEL = "vox256"

FSRT_MODEL_PATHS = {
    "vox256": "module_fsrt/models/vox256.pt",
    "vox256_2Source": "module_fsrt/models/vox256_2Source.pt",
}

FSRT_KP_PATH = "module_fsrt/models/kp_detector.pt"
FSRT_CFG_PATHS = {
    "vox256": "module_fsrt/configs/vox256.yaml",
    "vox256_2Source": "module_fsrt/configs/vox256_2Source.yaml",
}
