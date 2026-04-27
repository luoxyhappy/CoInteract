from typing_extensions import Literal, TypeAlias

from ..models.wan_video_dit_s2v import WanS2VModel
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_vae import WanVideoVAE, WanVideoVAE38
from ..models.wav2vec import WanS2VAudioEncoder

model_loader_configs = [
    # (state_dict_keys_hash, state_dict_keys_hash_with_shape, model_names, model_classes, model_resource)
    (None, "966cffdcc52f9c46c391768b27637614", ["wan_video_dit"], [WanS2VModel], "civitai"),
    (None, "9c8818c2cbea55eca56c7b447df170da", ["wan_video_text_encoder"], [WanTextEncoder], "civitai"),
    (None, "1378ea763357eea97acdef78e65d6d96", ["wan_video_vae"], [WanVideoVAE], "civitai"),
    (None, "ccc42284ea13e1ad04693284c7a09be6", ["wan_video_vae"], [WanVideoVAE], "civitai"),
    (None, "e1de6c02cdac79f8b739f4d3698cd216", ["wan_video_vae"], [WanVideoVAE38], "civitai"),
    (None, "45d5aec5a0864842822690d105ddb8ae", ["wans2v_audio_encoder"], [WanS2VAudioEncoder], "civitai"),
]

huggingface_model_loader_configs = [
    ("MarianMTModel", "transformers.models.marian.modeling_marian", "translator", None),
]

patch_model_loader_configs = []

preset_models_on_huggingface = {}
preset_models_on_modelscope = {}

Preset_model_id: TypeAlias = Literal["placeholder"]
