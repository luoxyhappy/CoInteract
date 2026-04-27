from transformers import Wav2Vec2Config, Wav2Vec2Model
from transformers.modeling_outputs import BaseModelOutput

from diffsynth.audio_analysis.torch_utils import linear_interpolation

# the implementation of Wav2Vec2Model is borrowed from
# https://github.com/huggingface/transformers/blob/HEAD/src/transformers/models/wav2vec2/modeling_wav2vec2.py
# initialize our encoder with the pre-trained wav2vec 2.0 weights.
class Wav2Vec2Model(Wav2Vec2Model):
    def __init__(self):
        #from chinese wav2vec config.json 
        config_dict = {
            "activation_dropout": 0.1,   #为什么要注释?test
            "adapter_kernel_size": 3,
            "adapter_stride": 2,
            "add_adapter": False,
            "apply_spec_augment": True,
            "architectures": [
                "Wav2Vec2ForPreTraining"
            ],
            "attention_dropout": 0.1,
            "bos_token_id": 1,
            "classifier_proj_size": 256,
            "codevector_dim": 256,
            "contrastive_logits_temperature": 0.1,
            "conv_bias": False,
            "conv_dim": [
                512,
                512,
                512,
                512,
                512,
                512,
                512
            ],
            "conv_kernel": [
                10,
                3,
                3,
                3,
                3,
                2,
                2
            ],
            "conv_stride": [
                5,
                2,
                2,
                2,
                2,
                2,
                2
            ],
            "ctc_loss_reduction": "sum",
            "ctc_zero_infinity": False,
            "diversity_loss_weight": 0.1,
            "do_stable_layer_norm": False,
            "eos_token_id": 2,
            "feat_extract_activation": "gelu",
            "feat_extract_norm": "group",
            "feat_proj_dropout": 0.0,
            "feat_quantizer_dropout": 0.0,
            "final_dropout": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-05,
            "layerdrop": 0.1,
            "mask_feature_length": 10,
            "mask_feature_min_masks": 0,
            "mask_feature_prob": 0.0,
            "mask_time_length": 10,
            "mask_time_min_masks": 2,
            "mask_time_prob": 0.05,
            "model_type": "wav2vec2",
            "num_adapter_layers": 3,
            "num_attention_heads": 12,
            "num_codevector_groups": 2,
            "num_codevectors_per_group": 320,
            "num_conv_pos_embedding_groups": 16,
            "num_conv_pos_embeddings": 128,
            "num_feat_extract_layers": 7,
            "num_hidden_layers": 12,
            "num_negatives": 100,
            "output_hidden_size": 768,
            "pad_token_id": 0,
            "proj_codevector_dim": 256,
            "tdnn_dilation": [
                1,
                2,
                3,
                1,
                1
            ],
            "tdnn_dim": [
                512,
                512,
                512,
                512,
                1500
            ],
            "tdnn_kernel": [
                5,
                3,
                3,
                1,
                1
            ],
            "torch_dtype": "float32",
            "transformers_version": "4.16.2",
            "use_weighted_layer_sum": False,
            "vocab_size": 32,
            "xvector_output_dim": 512,
            }
        model_config = Wav2Vec2Config(**config_dict)
        model_config._attn_implementation = "eager"

        super().__init__(model_config)

    def forward(
        self,
        input_values,
        seq_len,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        #self.config._attn_implementation = "eager"
        self.config.output_attentions = True

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, ) + encoder_outputs[1:]
        #encoder_outputs.hidden_states 是一个tuple类型,如果transformer有12层,则这个元素有13个,0号元素为input,去掉后就是其余12层的特征
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


    def feature_extract(
        self,
        input_values,
        seq_len,
    ):
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)

        return extract_features

    def encode(
        self,
        extract_features,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        self.config.output_attentions = True

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )
            

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, ) + encoder_outputs[1:]
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    @staticmethod
    def state_dict_converter():
        return AudioEncoderStateDictConverter()

class AudioEncoderStateDictConverter():
    def __init__(self):
        pass
    #borrow from LiveAIDiffStudio
    def from_civitai(self, state_dict):
        #state_dict = {'model.' + k: v for k, v in state_dict.items()}
        ignore_list = ["project_hid.bias", "project_hid.weight", "project_q.bias", "project_q.weight", "quantizer.codevectors", "quantizer.weight_proj.bias", "quantizer.weight_proj.weight"] 

        state_dict = {k.replace('wav2vec2.', '') : v for k, v in state_dict.items() if k not in ignore_list }
        return state_dict

