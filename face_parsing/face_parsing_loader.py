import torch
import numpy as np

from .face_parsing_model import BiSeNet


def load_face_parser_model(base_dir: str, face_parser_checkpoint_name: str, cpu=False):
    print("Loading face_parser model")
    face_parser = BiSeNet(n_classes=19)
    face_parser_path = f"{base_dir}/face_parsing/{face_parser_checkpoint_name}"

    mean = torch.Tensor(np.array([0.485, 0.456, 0.406], dtype=np.float32)).view(
        1, 3, 1, 1
    )
    std = torch.Tensor(np.array([0.229, 0.224, 0.225], dtype=np.float32)).view(
        1, 3, 1, 1
    )

    if not cpu:
        face_parser.cuda()
        face_parser.load_state_dict(torch.load(face_parser_path))
        face_parser.mean = mean.cuda()
        face_parser.std = std.cuda()
    else:
        face_parser.load_state_dict(
            torch.load(face_parser_path, map_location=torch.device("cpu"))
        )
        face_parser.mean = mean
        face_parser.std = std
    print("face_parser model loaded!")

    return face_parser
