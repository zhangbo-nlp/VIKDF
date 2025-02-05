# coding=utf-8

""" PyTorch ZRIGF-2 model."""
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Union

import math
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, CLIPModel, LlamaForCausalLM, Blip2QFormerConfig
from transformers.models.clip.modeling_clip import (
    CLIPTextEmbeddings, CLIPVisionEmbeddings, CLIPAttention, CLIPMLP, CLIPEncoder
)
from transformers.utils import logging, ModelOutput

from model.configuration_zrigf_2 import ZRIGF2Config
from model.modeling_qformer import BertModel

logger = logging.get_logger(__name__)


# Copied from https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_vicuna_instruct.py
def concat_text_input_output(input_ids, input_atts, output_ids, output_atts):
    input_part_targets_len = []
    llm_tokens = {"input_ids": [], "attention_mask": []}
    for i in range(input_ids.size(0)):
        this_input_ones = input_atts[i].sum()
        input_part_targets_len.append(this_input_ones)
        llm_tokens['input_ids'].append(
            torch.cat([
                input_ids[i][:this_input_ones],
                output_ids[i][1:],
                input_ids[i][this_input_ones:]
            ])
        )
        llm_tokens['attention_mask'].append(
            torch.cat([
                input_atts[i][:this_input_ones],
                output_atts[i][1:],
                input_atts[i][this_input_ones:]
            ])
        )
    llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
    llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
    return llm_tokens, input_part_targets_len


# contrastive loss function, adapted from https://sachinruk.github.io/blog/2021-03-07-clip.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.ReLU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        attn_output, _ = self.multihead_attention(Q, K, V)
        x = self.norm1(Q + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


@dataclass
class ZRIGF2ModelOutput(ModelOutput):
    loss: Optional[Tuple[torch.FloatTensor]] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ["vision_outputs", "text_outputs", "qformer_outputs", "language_model_outputs"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )


@dataclass
class ZRIGF2ForConditionalGenerationOutput(ModelOutput):
    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    text_outputs: Optional[torch.FloatTensor] = None
    qformer_outputs: Optional[Tuple[torch.FloatTensor]] = None
    language_model_outputs: Optional[Tuple[torch.FloatTensor]] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ["vision_outputs", "text_outputs", "qformer_outputs", "language_model_outputs"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class ZRIGF2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ZRIGF2Config
    base_model_prefix = "zrigf"
    supports_gradient_checkpointing = True
    _no_split_modules = ["CLIPAttention", "LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _keep_in_fp32_modules = ["wo"]

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor

        if isinstance(module, CLIPTextEmbeddings):
            module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
            module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        elif isinstance(module, CLIPVisionEmbeddings):
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim ** -0.5 * factor)
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        elif isinstance(module, CLIPAttention):
            in_proj_std = (module.embed_dim ** -0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim ** -0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        elif isinstance(module, CLIPMLP):
            in_proj_std = (
                    (module.config.hidden_size ** -0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            )
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)

        elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
            factor = self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=factor)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, CLIPEncoder):
            module.gradient_checkpointing = value


class ZRIGF2ForConditionalGeneration(ZRIGF2PreTrainedModel):
    config_class = ZRIGF2Config

    _keys_to_ignore_on_load_unexpected = [r"clip_vision_model", r"clip_visual_projection", r"query_text_projection",
                                       r"query_visual_projection", r"logit_scale", r"visual_tb", r"visual_decoder",
                                       r"text_tb", r"text_decoder", r"query_llm_projection"]

    def __init__(
            self,
            config: ZRIGF2Config = None,
            clip_model: CLIPModel = None,
            qformer: BertModel = None,
            language_model: LlamaForCausalLM = None,
    ):
        if config is None and (clip_model is None or qformer is None or language_model is None):
            raise ValueError("Either a configuration or models have to be provided")

        if config is None:
            config = ZRIGF2Config.from_configs(clip_model.config, qformer.config, language_model.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"config: {config} has to be of type {self.config_class}")

        super().__init__(config)
        self.pretraining = getattr(config, "pretraining", True)

        if clip_model is None:
            clip_model = CLIPModel(config.clip_config)

        if qformer is None:
            qformer = BertModel(config.qformer_config)

        if language_model is None:
            language_model = LlamaForCausalLM(config.llm_config)

        self.clip_text_model = clip_model.text_model
        self.clip_text_projection = clip_model.text_projection
        if self.pretraining:
            self.clip_vision_model = clip_model.vision_model
            self.clip_visual_projection = clip_model.visual_projection

        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.query_tokens.data.normal_(mean=0.0, std=config.initializer_range)
        self.qformer = qformer

        if self.pretraining:
            self.query_text_projection = nn.Linear(config.qformer_config.hidden_size, config.query_projection_dim)
            self.query_visual_projection = nn.Linear(config.qformer_config.hidden_size, config.query_projection_dim)

            self.logit_scale = nn.Parameter(torch.tensor(config.clip_config.logit_scale_init_value))

        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.llm_config.hidden_size)
        # Update _tied_weights_keys using the base model used.
        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in language_model._tied_weights_keys]
        self.language_model = language_model

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.clip_text_model.config = self.config.clip_config.text_config
        if self.pretraining:
            self.clip_vision_model.config = self.config.clip_config.vision_config
        self.qformer.config = self.config.qformer_config

        if self.pretraining:
            self.unchanged_ratio = getattr(config, "unchanged_ratio", 0.)
            self.num_channels = config.clip_config.vision_config.num_channels
            self.visual_tb = TransformerBlock(
                hidden_size=config.qformer_config.hidden_size,
                num_heads=4,
            )
            self.visual_decoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=config.qformer_config.hidden_size,
                    out_channels=config.encoder_stride ** 2 * self.num_channels,
                    kernel_size=1,
                ),
                nn.PixelShuffle(config.encoder_stride),
            )

            self.masked_ratio = getattr(config, "masked_ratio", 0.15)
            self.text_tb = TransformerBlock(
                hidden_size=config.qformer_config.hidden_size,
                num_heads=4,
            )
            self.text_decoder = nn.Linear(config.qformer_config.hidden_size, config.clip_config.text_config.vocab_size)

            self.query_llm_projection = nn.Linear(config.llm_config.hidden_size, config.qformer_config.hidden_size)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _preprocess_accelerate(self):
        r"""
        Some pre-processing hacks to make the model `accelerate` compatible. Check
        https://github.com/huggingface/transformers/pull/21707 for more details.
        """
        hf_device_map = self.hf_device_map

        if len(hf_device_map) > 1 and "language_model" not in hf_device_map and torch.cuda.device_count() > 1:
            # warn users about unexpected behavior when using multi-GPU + BLIP-2 + `accelerate`.
            logger.warning(
                "The `language_model` is not in the `hf_device_map` dictionary and you are running your script"
                " in a multi-GPU environment. this may lead to unexpected behavior when using `accelerate`."
                " Please pass a `device_map` that contains `language_model` to remove this warning."
                " Please refer to https://github.com/huggingface/blog/blob/main/accelerate-large-models.md for"
                " more details on creating a `device_map` for large models.",
            )

        if hasattr(self.language_model, "_hf_hook"):
            self.language_model._hf_hook.io_same_device = True  # For `generate` compatibility

    def forward(
            self,
            text_input_ids: torch.LongTensor,
            i2t_llm_input_ids: torch.LongTensor,
            bias: int = 3,
            pixel_values: Optional[torch.FloatTensor] = None,
            bool_masked_pos: Optional[torch.Tensor] = None,
            masked_indices: Optional[torch.Tensor] = None,
            masked_text_labels: Optional[torch.LongTensor] = None,
            text_attention_mask: Optional[torch.LongTensor] = None,
            i2t_llm_attention_mask: Optional[torch.LongTensor] = None,
            t2i_llm_input_ids: Optional[torch.LongTensor] = None,
            t2i_llm_attention_mask: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ZRIGF2ForConditionalGenerationOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        loss = 0

        # forward the texts through the text encoder of clip model,
        # to get text embeddings of shape (batch_size, seq_len, hidden_size)
        clip_text_outputs = self.clip_text_model(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        clip_text_embeds = self.clip_text_projection(clip_text_outputs[0])

        # forward the query tokens through the QFormer, using the text embeddings for cross-attention
        query_tokens = self.query_tokens.expand(clip_text_embeds.shape[0], -1, -1)
        query_text_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=clip_text_embeds,
            encoder_attention_mask=text_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if pixel_values is not None:
            # forward the images through the vision encoder of clip model,
            # to get image embeddings of shape (batch_size, seq_len, hidden_size)
            clip_vision_outputs = self.clip_vision_model(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            clip_image_embeds = self.clip_visual_projection(clip_vision_outputs[0])  # last_hidden_state

            clip_image_attention_mask = torch.ones(
                clip_image_embeds.size()[:-1], dtype=torch.long, device=clip_image_embeds.device
            )
            query_vision_outputs = self.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=clip_image_embeds,
                encoder_attention_mask=clip_image_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            # Text-Image Matching Module
            image_embeds = self.query_visual_projection(query_vision_outputs[1])  # pooler_output
            text_embeds = self.query_text_projection(query_text_outputs[1])

            # normalized features
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

            # cosine similarity as logits
            logit_scale = self.logit_scale.exp()
            logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale

            loss_clip = clip_loss(logits_per_text)
            loss += loss_clip * 0.5

            # Text-Assisted Masked Image Modeling Module
            if bool_masked_pos is not None:
                # unchanged_ratio of the time, we keep the masked input tokens unchanged
                bool_masked_pos_rand = torch.bernoulli(
                    torch.full(bool_masked_pos.shape, 1 - self.unchanged_ratio, device=bool_masked_pos.device)
                ).bool() & bool_masked_pos

                size = self.config.clip_config.vision_config.image_size // self.config.clip_config.vision_config.patch_size
                bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
                bool_masked_pos_rand = bool_masked_pos_rand.reshape(-1, size, size)
                bool_mask = (
                    bool_masked_pos.repeat_interleave(self.config.clip_config.vision_config.patch_size, 1)
                    .repeat_interleave(self.config.clip_config.vision_config.patch_size, 2)
                    .unsqueeze(1)
                    .contiguous()
                )
                bool_mask_rand = (
                    bool_masked_pos_rand.repeat_interleave(self.config.clip_config.vision_config.patch_size, 1)
                    .repeat_interleave(self.config.clip_config.vision_config.patch_size, 2)
                    .unsqueeze(1)
                    .contiguous()
                )

                masked_pixel_values = pixel_values.clone() * (1 - bool_mask_rand)

                mask_clip_vision_outputs = self.clip_vision_model(
                    pixel_values=masked_pixel_values,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                mask_clip_image_embeds = self.clip_visual_projection(mask_clip_vision_outputs[0])

                # Call TransformerBlock and get fusion features
                sequence_output = self.visual_tb(mask_clip_image_embeds, query_text_outputs[0], query_text_outputs[0])
                sequence_output = sequence_output[:, 1:]

                # Reshape to (batch_size, num_channels, height, width)
                batch_size, sequence_length, num_channels = sequence_output.shape
                height = width = math.floor(sequence_length ** 0.5)
                sequence_output = (
                    sequence_output.permute(0, 2, 1).contiguous()
                    .reshape(batch_size, num_channels, height, width)
                )
                # Reconstruct pixel values
                reconstructed_pixel_values = self.visual_decoder(sequence_output)

                reconstruction_loss = nn.functional.l1_loss(pixel_values, reconstructed_pixel_values, reduction="none")
                loss_recon = (reconstruction_loss * bool_mask).sum() / (bool_mask.sum() + 1e-5) / self.num_channels
                loss += loss_recon * 0.2

            # Image-Assisted Masked Text Modeling Module
            if masked_indices is not None:
                indices_replaced = torch.bernoulli(
                    torch.full(masked_text_labels.shape, 1 - self.unchanged_ratio, device=bool_masked_pos.device)
                ).bool() & masked_indices

                masked_text_input_ids = text_input_ids.clone()
                masked_text_input_ids[indices_replaced] = 49407  # tokenizer.unk_token_id

                masked_clip_text_outputs = self.clip_text_model(
                    input_ids=masked_text_input_ids,
                    attention_mask=text_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                masked_clip_text_embeds = self.clip_text_projection(masked_clip_text_outputs[0])

                sequence_output = self.text_tb(masked_clip_text_embeds, query_vision_outputs[0], query_vision_outputs[0])
                prediction_scores = self.text_decoder(sequence_output)

                loss_fct = CrossEntropyLoss()
                loss_mlm = loss_fct(
                    prediction_scores.view(-1, self.config.clip_config.text_config.vocab_size),
                    masked_text_labels.view(-1)
                )
                loss += loss_mlm * 0.2

        # I2T: use the language model, conditioned on the query outputs
        query_text_output = query_text_outputs[0]
        query_embeds = self.language_projection(query_text_output)
        query_attention_mask = torch.ones(
            query_embeds.size()[:-1], dtype=torch.long, device=query_embeds.device
        )

        expected_device = query_embeds.device
        i2t_llm_embeds = self.language_model.get_input_embeddings()(i2t_llm_input_ids).to(expected_device)
        i2t_input_embeds = torch.cat([i2t_llm_embeds[:, :bias, :], query_embeds, i2t_llm_embeds[:, bias:, :]], dim=1)

        if i2t_llm_attention_mask is None:
            i2t_llm_attention_mask = torch.ones_like(i2t_llm_input_ids)
        i2t_llm_attention_mask = i2t_llm_attention_mask.to(expected_device)
        i2t_attention_mask = torch.cat(
            [i2t_llm_attention_mask[:, :bias], query_attention_mask, i2t_llm_attention_mask[:, bias:]], dim=1
        )

        i2t_outputs = self.language_model(
            inputs_embeds=i2t_input_embeds,
            attention_mask=i2t_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        i2t_logits = i2t_outputs.logits if return_dict else i2t_outputs[0]
        # we compute the loss here since we need to take into account the sequence length of the query embeds
        i2t_labels = labels.to(i2t_logits.device)
        i2t_logits = i2t_logits[:, -i2t_labels.size(1):, :]
        # Shift so that tokens < n predict n
        shift_logits = i2t_logits[..., :-1, :].contiguous()
        shift_labels = i2t_labels[..., 1:].contiguous().to(i2t_logits.device)

        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction="mean")
        loss_i2t = loss_fct(shift_logits.view(-1, self.language_model.config.vocab_size), shift_labels.view(-1))
        loss += loss_i2t

        # T2I: use the language model, generate the query outputs
        if t2i_llm_input_ids is not None:
            expected_device = query_embeds.device
            t2i_llm_embeds = self.language_model.get_input_embeddings()(t2i_llm_input_ids).to(expected_device)
            if t2i_llm_attention_mask is None:
                t2i_llm_attention_mask = torch.ones_like(t2i_llm_input_ids)
            t2i_llm_attention_mask = t2i_llm_attention_mask.to(expected_device)

            t2i_input_embeds = []
            t2i_attention_masks = []
            llm_ones_counts = t2i_llm_attention_mask.sum(dim=1)

            for i, ones_count in enumerate(llm_ones_counts):
                t2i_input_embeds.append(
                    torch.cat([
                        t2i_llm_embeds[i, :ones_count],
                        query_embeds[i],
                        t2i_llm_embeds[i, ones_count:]
                    ], dim=0)
                )

                t2i_attention_masks.append(
                    torch.cat([
                        t2i_llm_attention_mask[i, :ones_count],
                        query_attention_mask[i],
                        t2i_llm_attention_mask[i, ones_count:]
                    ], dim=0)
                )

            # stack the results
            t2i_input_embeds = torch.stack(t2i_input_embeds)
            t2i_attention_masks = torch.stack(t2i_attention_masks)

            t2i_outputs = self.language_model.model(
                inputs_embeds=t2i_input_embeds,
                attention_mask=t2i_attention_masks,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            query_len = query_embeds.size(1)
            query_llm_output = torch.stack(
                [t2i_outputs[0][i, ones_count:ones_count+query_len] for i, ones_count in enumerate(llm_ones_counts-1)]
            )

            loss_t2i = nn.functional.mse_loss(self.query_llm_projection(query_llm_output), query_text_output)
            loss += loss_t2i * 0.5

        logits = i2t_logits
        outputs = i2t_outputs

        if not return_dict:
            output = (logits, clip_text_outputs, query_text_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return ZRIGF2ForConditionalGenerationOutput(
            loss=loss,
            logits=logits,
            text_outputs=clip_text_outputs,
            qformer_outputs=query_text_outputs,
            language_model_outputs=outputs,
        )

    @torch.no_grad()
    def generate(
            self,
            text_input_ids: torch.LongTensor,
            llm_input_ids: torch.LongTensor,
            text_attention_mask: Optional[torch.LongTensor] = None,
            llm_attention_mask: Optional[torch.LongTensor] = None,
            bias: int = 4,
            **generate_kwargs,
    ) -> torch.LongTensor:
        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()

        batch_size = text_input_ids.shape[0]

        clip_text_outputs = self.clip_text_model(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            return_dict=True,
        )
        clip_text_embeds = self.clip_text_projection(clip_text_outputs[0])
        query_tokens = self.query_tokens.expand(clip_text_embeds.shape[0], -1, -1)
        query_text_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=clip_text_embeds,
            encoder_attention_mask=text_attention_mask,
            return_dict=True,
        )
        query_text_output = query_text_outputs[0]

        query_embeds = self.language_projection(query_text_output)
        query_attention_mask = torch.ones(
            query_embeds.size()[:-1], dtype=torch.long, device=query_embeds.device
        )
        if llm_input_ids is None:
            llm_input_ids = (
                torch.LongTensor([[self.config.llm_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(clip_text_embeds.device)
            )
        if llm_attention_mask is None:
            llm_attention_mask = torch.ones_like(llm_input_ids)
        expected_device = query_attention_mask.device
        attention_mask = torch.cat([query_attention_mask, llm_attention_mask.to(expected_device)], dim=1)

        # concatenate query embeddings with prompt embeddings
        llm_embeds = self.language_model.get_input_embeddings()(llm_input_ids).to(query_embeds.device)
        inputs_embeds = torch.cat([llm_embeds[:, :bias, :], query_embeds, llm_embeds[:, bias:, :]], dim=1)

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        if isinstance(outputs, torch.Tensor):
            outputs[outputs == 0] = 2
        else:
            outputs.sequences[outputs.sequences == 0] = 2

        return outputs

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported
        # for composite models
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_clip_qformer_llm_pretrained(
            cls,
            clip_model_name_or_path: str,
            qformer_name_or_path: str,
            language_model_name_or_path: str,
            *model_args,
            **kwargs,
    ) -> PreTrainedModel:

        clip_model = CLIPModel.from_pretrained(clip_model_name_or_path)

        qformer_config = Blip2QFormerConfig.from_pretrained(qformer_name_or_path)
        qformer_config.encoder_hidden_size = clip_model.config.projection_dim
        qformer = BertModel.from_pretrained(qformer_name_or_path, config=qformer_config)

        language_model = LlamaForCausalLM.from_pretrained(language_model_name_or_path, *model_args, **kwargs)

        # instantiate config with corresponding kwargs
        config = ZRIGF2Config.from_configs(clip_model.config, qformer.config, language_model.config)

        # init model
        model = cls(config=config, clip_model=clip_model, qformer=qformer, language_model=language_model)

        return model
