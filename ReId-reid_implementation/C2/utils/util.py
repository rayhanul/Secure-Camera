import numpy as np
import torch


def objects_to_tensor(objects):
    # Each procesed_image is a list; convert to tensor and stack
    tensors = []
    for object in objects:
        if isinstance(object["processed_image"], list):
            try:
                tensor = torch.tensor(object["processed_image"])
            except ValueError:
                continue
        elif isinstance(object["processed_image"], np.ndarray):
            tensor = torch.from_numpy(object["processed_image"])
        elif isinstance(object["processed_image"], torch.Tensor):
            tensor = object["processed_image"]
        else:
            raise TypeError(f"Unsupported type: {type(object['processed_image'])}")

        tensors.append(tensor)

    if not tensors:
        raise ValueError("No valid tensors found")

    return torch.stack(tensors)
