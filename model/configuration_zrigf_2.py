# coding=utf-8

""" ZRIGF-2 model configuration"""

from transformers import Blip2QFormerConfig, CLIPConfig, LlamaConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class ZRIGF2Config(PretrainedConfig):
    model_type = "zrigf-2"

    def __init__(
            self,
            clip_config=None,
            qformer_config=None,
            llm_config=None,
            num_query_tokens=32,
            query_projection_dim=256,
            **kwargs
    ):
        super().__init__(**kwargs)

        if clip_config is None:
            clip_config = {}
            logger.info("clip_config is None. initializing the CLIPConfig with default values.")

        if qformer_config is None:
            qformer_config = {}
            logger.info("qformer_config is None. Initializing the Blip2QFormerConfig with default values.")

        if llm_config is None:
            llm_config = {}
            logger.info("llm_config is None. Initializing the LlamaConfig with default values.")

        self.clip_config = CLIPConfig(**clip_config)
        self.qformer_config = Blip2QFormerConfig(**qformer_config)
        self.llm_config = LlamaConfig(**llm_config)

        self.tie_word_embeddings = self.llm_config.tie_word_embeddings
        self.is_encoder_decoder = self.llm_config.is_encoder_decoder

        self.num_query_tokens = num_query_tokens
        self.query_projection_dim = query_projection_dim
        self.qformer_config.encoder_hidden_size = self.clip_config.projection_dim

        self.encoder_stride = self.clip_config.vision_config.patch_size

        self.initializer_factor = 1.0
        self.initializer_range = 0.02


    @classmethod
    def from_configs(
            cls,
            clip_config: CLIPConfig,
            qformer_config: Blip2QFormerConfig,
            llm_config: LlamaConfig = None,
            **kwargs,
    ):
        return cls(
            clip_config=clip_config.to_dict(),
            qformer_config=qformer_config.to_dict(),
            llm_config=llm_config.to_dict() if llm_config is not None else None,
            **kwargs,
        )
