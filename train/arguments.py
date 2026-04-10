import os
from typing import List, Optional
from dataclasses import dataclass, field

from FlagEmbedding.abc.finetune.embedder import (
    AbsEmbedderTrainingArguments,
)


def default_target_modules() -> List[int]:
    return ['v_proj', 'q_proj', 'k_proj', 'gate_proj', 'down_proj', 'o_proj', 'up_proj']


@dataclass
class QueryEncoderOnlyEmbedderModelArguments:
    """
    Model argument class for Query Encoder Only Embedder.
    """
    config_name_query: str = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name."}
    )
    model_name_or_path_query: str = field(
        default=None,
        metadata={"help": "The model checkpoint for initialization."}
    )
    trust_remote_code_query: bool = field(
        default=False,
        metadata={"help": "Trust remote code"}
    )


@dataclass
class DocDecoderOnlyEmbedderModelArguments:
    """
    Model argument class for Doc Decoder Only Embedder.
    """
    config_name_doc: str = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name."}
    )
    peft_model_path: str = field(
        default='', metadata={"help": "The peft model checkpoint for initialization."}
    )
    model_name_or_path_doc: str = field(
        default=None,
        metadata={"help": "The model checkpoint for initialization."}
    )
    trust_remote_code_doc: bool = field(
        default=False,
        metadata={"help": "Trust remote code"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "If passed, will use LORA (low-rank parameter-efficient training) to train the model."}
    )
    lora_rank: int = field(
        default=64,
        metadata={"help": "The rank of lora."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": "The alpha parameter of lora."}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout rate of lora modules."}
    )
    target_modules: List[str] = field(
        default_factory=default_target_modules,
        metadata={"help": "The target modules to apply LORA."}
    )
    use_flash_attn: bool = field(
        default=False,
        metadata={"help": "If passed, will use flash attention to train the model."}
    )
    use_slow_tokenizer: bool = field(
        default=False,
        metadata={"help": "If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library)."}
    )
    # low_cpu_mem_usage: bool = field(
    #     default=False,
    #     metadata={"help": "It is an option to create the model as an empty shell,"
    #                       "then only materialize its parameters when the pretrained weights are loaded."
    #                       "If passed, LLM loading time and RAM consumption will be benefited."}
    # )
    from_peft: str = field(
        default=None
    )
    modules_to_save: str = field(
        default=None
    )
    raw_peft: str = field(
        default=None
    )

    additional_special_tokens: Optional[str] = field(
        default=None,
        metadata={"help": "additional special tokens", "nargs": "+"}
    )
    
    save_merged_lora_model: bool = field(
        default=False,
        metadata={"help": "If passed, will merge the lora modules and save the entire model."}
    )

    only_merge_lora_model: bool = field(
        default=False,
        metadata={"help": "If passed, will only merge the lora modules and save the entire model."}
    )


@dataclass
class AsymmetricEmbedderModelArguments(
    QueryEncoderOnlyEmbedderModelArguments,
    DocDecoderOnlyEmbedderModelArguments
):
    """
    Model argument class for Asymmetric Embedder.
    """
    cache_dir: str = field(
        default=None,
        metadata={"help": "Where do you want to store the pre-trained models downloaded from s3."}
    )
    token: str = field(
        default_factory=lambda: os.getenv('HF_TOKEN', None),
        metadata={"help": "The token to use when accessing the model."}
    )


@dataclass
class AsymmetricEmbedderTrainingArguments(AbsEmbedderTrainingArguments):
    """
    Training argument class for Asymmetric Embedder.
    """
    fix_doc_encoder: bool = field(default=False, metadata={"help": "Freeze the parameters of query encoder and doc encoder"})
    use_mrl: bool=field(
        default=False,
        metadata={'help':'whether to use mrl'}
    )
    mrl_dims: str=field(
        default='128, 256, 512, 768, 1024, 1280, 1536, 1792',
        metadata={'help':'mrl dims'}
    )
    k: float=field(
        default=1,
        metadata={'help':'the threshold for in-batch false negative mask'}
    )
    kd_loss_type: str = field(default=None, metadata={"help": "the loss type for knowledge distillation. Available options: kl_div, m3_kd_loss. Default: kl_div.", "choices": ['contrastive', 'mse', 'cossim', 'cossim_and_mse', 'contrastive_and_mse']})
