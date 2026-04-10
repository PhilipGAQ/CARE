import os
import torch
import logging
from typing import Tuple
from transformers import (
    AutoModel, AutoConfig,
    AutoTokenizer, PreTrainedTokenizer
)
from huggingface_hub import snapshot_download

from FlagEmbedding.abc.finetune.embedder import (
    AbsEmbedderRunner, AbsEmbedderModel,
    AbsEmbedderDataArguments, EmbedderTrainerCallbackForDataRefresh
)
from modeling import AsymmetricEmbedderModel
from trainer import AsymmetricEmbedderTrainer
from arguments import AsymmetricEmbedderModelArguments, AsymmetricEmbedderTrainingArguments
from dataset import AsymmetricEmbedderCollator, AsymmetricEmbedderSameDatasetCollator, AsymmetricEmbedderSameDatasetCollator_distill

from load_model import get_model, save_merged_model

logger = logging.getLogger(__name__)


class AsymmetricEmbedderRunner(AbsEmbedderRunner):
    def __init__(
        self,
        model_args: AsymmetricEmbedderModelArguments,
        data_args: AbsEmbedderDataArguments,
        training_args: AsymmetricEmbedderTrainingArguments
    ):
        super().__init__(model_args, data_args, training_args)
        self.model_args: AsymmetricEmbedderModelArguments
        self.data_args: AbsEmbedderDataArguments
        self.training_args: AsymmetricEmbedderTrainingArguments

    def load_tokenizer_and_model(self) -> Tuple[PreTrainedTokenizer, AbsEmbedderModel]:
        """Load the tokenizer and model.

        Returns:
            Tuple[PreTrainedTokenizer, AbsEmbedderModel]: Tokenizer and model instances.
        """
        query_tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path_query,
            cache_dir=self.model_args.cache_dir,
            trust_remote_code=self.model_args.trust_remote_code_query
        )
        query_encoder = AutoModel.from_pretrained(
            self.model_args.model_name_or_path_query,
            cache_dir=self.model_args.cache_dir,
            trust_remote_code=self.model_args.trust_remote_code_query
        )
        
        doc_tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path_doc,
            token=self.model_args.token,
            cache_dir=self.model_args.cache_dir,
            use_fast=False,
            add_eos_token=True,
            trust_remote_code=self.model_args.trust_remote_code_doc,
        )
        if doc_tokenizer.pad_token is None:
            if doc_tokenizer.unk_token is not None:
                doc_tokenizer.pad_token = doc_tokenizer.unk_token
                doc_tokenizer.pad_token_id = doc_tokenizer.unk_token_id
            else:
                doc_tokenizer.pad_token = doc_tokenizer.eos_token
                doc_tokenizer.pad_token_id = doc_tokenizer.eos_token_id
        doc_tokenizer.padding_side = 'left'
        
        resize = False
        if self.model_args.additional_special_tokens is not None:
            special_tokens_dict = {'additional_special_tokens': self.model_args.additional_special_tokens}
            add_num = doc_tokenizer.add_special_tokens(special_tokens_dict)
            if add_num > 0:
                resize = True
                logger.info(f"Add {add_num} special tokens to the tokenizer. Special tokens: {self.model_args.additional_special_tokens}")
            else:
                logger.warning(f"Special tokens {self.model_args.additional_special_tokens} already exists in the tokenizer.")
        doc_encoder = get_model(self.model_args, self.training_args.output_dir, resize, len(doc_tokenizer))

        num_labels = 1
        query_config = AutoConfig.from_pretrained(
            self.model_args.config_name_query if self.model_args.config_name_query else self.model_args.model_name_or_path_query,
            num_labels=num_labels,
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code_query,
        )
        logger.info('Query Encoder Config: %s', query_config)
        
        doc_config = AutoConfig.from_pretrained(
            self.model_args.config_name_doc if self.model_args.config_name_doc else self.model_args.model_name_or_path_doc,
            num_labels=num_labels,
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code_doc,
        )
        logger.info('Doc Encoder Config: %s', doc_config)

        base_model = {
            "query_encoder": query_encoder,
            "doc_encoder": doc_encoder,
        }
        tokenizer = {
            "query_tokenizer": query_tokenizer,
            "doc_tokenizer": doc_tokenizer
        }


        mrl_dims=self.training_args.mrl_dims.strip().split(',')
        mrl_dims=[int(dim.strip()) for dim in mrl_dims]

        model = AsymmetricEmbedderModel(
            base_model=base_model,
            tokenizer=tokenizer,
            negatives_cross_device=self.training_args.negatives_cross_device,
            temperature=self.training_args.temperature,
            sub_batch_size=self.training_args.sub_batch_size,
            kd_loss_type=self.training_args.kd_loss_type,
            sentence_pooling_method=self.training_args.sentence_pooling_method,
            normalize_embeddings=self.training_args.normalize_embeddings,
            use_mrl=self.training_args.use_mrl,
            mrl_dims=mrl_dims,
            k=self.training_args.k,
        )

        if self.training_args.gradient_checkpointing:
            model.enable_input_require_grads()

        if self.training_args.fix_position_embedding:
            for k, v in model.named_parameters():
                if "position_embeddings" in k:
                    logging.info(f"Freeze the parameters for {k}")
                    v.requires_grad = False
        
        if self.training_args.fix_doc_encoder:
            logging.info('fix doc encoder')
            for k, v in model.named_parameters():
                if "doc_encoder" in k:
                    v.requires_grad = False

        self.query_tokenizer = query_tokenizer
        self.doc_tokenizer = doc_tokenizer
        
        return tokenizer, model

    def load_data_collator(self):
        """Loads the appropriate data collator.

        Returns:
            AbsEmbedderCollator: Loaded data collator.
        """
        if self.data_args.same_dataset_within_batch:
            EmbedCollator = AsymmetricEmbedderSameDatasetCollator_distill
        else:
            EmbedCollator = AsymmetricEmbedderCollator

        data_collator = EmbedCollator(
            tokenizer=None,
            query_tokenizer=self.query_tokenizer,
            doc_tokenizer=self.doc_tokenizer,
            query_max_len=self.data_args.query_max_len,
            passage_max_len=self.data_args.passage_max_len,
            sub_batch_size=self.training_args.sub_batch_size,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            padding=True,
            return_tensors="pt"
        )
        return data_collator

    def load_trainer(self) -> AsymmetricEmbedderTrainer:
        """Load the trainer.
        """
        trainer = AsymmetricEmbedderTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
        )
        if self.data_args.same_dataset_within_batch:
            trainer.add_callback(EmbedderTrainerCallbackForDataRefresh(self.train_dataset))
        return trainer
