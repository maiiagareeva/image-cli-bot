from __future__ import annotations

from dataclasses import dataclass

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)


def _resolve_lavis_model_type(
    blip2_model_id,
    lavis_model_type,
):
    if lavis_model_type is not None:
        return lavis_model_type

    model_id = blip2_model_id.lower()
    if "coco" in model_id:
        return "coco"
    if "vitl" in model_id or "clip_l" in model_id:
        return "pretrain_vitL"
    return "pretrain"

def _image_size_for_model_type(model_type: str) -> int:
    if model_type == "coco":
        return 364
    return 224

@dataclass
class _TorchvisionImageProcessor:
    transform: transforms.Compose

    def __call__(self, image):
        return self.transform(image)


def build_blip2_image_processors(
    blip2_model_id ,
    lavis_model_type,
):
    model_type = _resolve_lavis_model_type(blip2_model_id, lavis_model_type)
    image_size = _image_size_for_model_type(model_type)
    normalize = transforms.Normalize(mean, std)

    train_processor = _TorchvisionImageProcessor(
        transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.5, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    )
    eval_processor = _TorchvisionImageProcessor(
        transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
    )

    return train_processor, eval_processor, model_type
