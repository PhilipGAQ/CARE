import os
import logging
from typing import Any, Dict, Union, List

import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import torch.distributed as dist
from FlagEmbedding.abc.finetune.embedder import AbsEmbedderModel, EmbedderOutput

logger = logging.getLogger(__name__)


class AsymmetricEmbedderModel(AbsEmbedderModel):
    """Embedder class for asymmetric embedding model.

    Args:
        base_model (AutoModel): The base model to train on.
        tokenizer (AutoTokenizer, optional): The tokenizer to use. Defaults to ``None``.
        negatives_cross_device (bool, optional): If True, will compute cross devices negative loss. Defaults to ``False``.
        temperature (float, optional): Temperature to control the scale of scores. Defaults to ``1.0``.
        sub_batch_size (int, optional): Sub-batch size during encoding. If negative, will not split to sub-batch.
            Defaults to ``-1``.
        kd_loss_type (str, optional): Type of knowledge distillation loss. Defaults to ``"kl_div"``.
        sentence_pooling_method (str, optional): Pooling method to get sentence embedding. Defaults to ``'cls'``.
        normalize_embeddings (bool, optional): If True, normalize the embedding vector. Defaults to ``False``.
    """
    TRANSFORMER_CLS = AutoModel
    
    def __init__(
        self,
        base_model: Dict[str, Any],
        tokenizer: AutoTokenizer = None,
        negatives_cross_device: bool = False,
        temperature: float = 1.0,
        sub_batch_size: int = -1,
        kd_loss_type: str = 'kl_div',
        sentence_pooling_method: str = 'cls',
        normalize_embeddings: bool = False,
        use_mrl: bool=True,
        mrl_dims=[],
        k=10.0
    ):
        super().__init__(
            base_model,
            tokenizer=tokenizer,
            negatives_cross_device=negatives_cross_device,
            temperature=temperature,
            sub_batch_size=sub_batch_size,
            kd_loss_type=kd_loss_type,
        )
        self.sentence_pooling_method = sentence_pooling_method
        self.normalize_embeddings = normalize_embeddings
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        
        self.model = None
        self.query_encoder = base_model["query_encoder"]
        self.doc_encoder = base_model["doc_encoder"]
        self.use_mrl = use_mrl
        self.mrl_dims=mrl_dims
        self.k = k
        print('use_mrl', use_mrl)
        print('mrl_dims', mrl_dims)
        print('k', k)

    def encode(self, features):
        pass

    def encode_queries(self, features):
        """Encode and get the embedding for queries.

        Args:
            features (Union[list, dict]): Features feed to the model.

        Returns:
            torch.Tensor: The embedding vectors.
        """
        if features is None:
            return None
        if not isinstance(features, list):
            if self.sub_batch_size is not None and self.sub_batch_size > 0:
                all_p_reps = []
                for i in range(0, len(features['attention_mask']), self.sub_batch_size):
                    end_inx = min(i + self.sub_batch_size, len(features['attention_mask']))
                    sub_features = {}
                    for k, v in features.items():
                        sub_features[k] = v[i:end_inx]
                    last_hidden_state = self.query_encoder(**sub_features, return_dict=True).last_hidden_state
                    p_reps = self._sentence_embedding(last_hidden_state, sub_features['attention_mask'])
                    all_p_reps.append(p_reps)
                all_p_reps = torch.cat(all_p_reps, 0).contiguous()
                if self.normalize_embeddings:
                    all_p_reps = torch.nn.functional.normalize(all_p_reps, dim=-1)
                return all_p_reps.contiguous()
            else:
                last_hidden_state = self.query_encoder(**features, return_dict=True).last_hidden_state
                all_p_reps = self._sentence_embedding(last_hidden_state, features['attention_mask'])
                if self.normalize_embeddings:
                    all_p_reps = torch.nn.functional.normalize(all_p_reps, dim=-1)
                return all_p_reps.contiguous()
        else:
            all_p_reps = []
            for sub_features in features:
                last_hidden_state = self.query_encoder(**sub_features, return_dict=True).last_hidden_state
                p_reps = self._sentence_embedding(last_hidden_state, sub_features['attention_mask'])
                all_p_reps.append(p_reps)
            all_p_reps = torch.cat(all_p_reps, 0).contiguous()
            if self.normalize_embeddings:
                all_p_reps = torch.nn.functional.normalize(all_p_reps, dim=-1)
            return all_p_reps.contiguous()

    def encode_corpus(self, features):
        """Encode and get the embedding for corpus.

        Args:
            features (Union[list, dict]): Features feed to the model.

        Returns:
            torch.Tensor: The embedding vectors.
        """
        if features is None:
            return None
        if not isinstance(features, list):
            if self.sub_batch_size is not None and self.sub_batch_size > 0:
                all_p_reps = []
                for i in range(0, len(features['attention_mask']), self.sub_batch_size):
                    end_inx = min(i + self.sub_batch_size, len(features['attention_mask']))
                    sub_features = {}
                    for k, v in features.items():
                        sub_features[k] = v[i:end_inx]
                    last_hidden_state = self.doc_encoder(**sub_features, return_dict=True).last_hidden_state
                    p_reps = self._sentence_embedding(last_hidden_state, sub_features['attention_mask'], sentence_pooling_method="last_token")
                    all_p_reps.append(p_reps)
                all_p_reps = torch.cat(all_p_reps, 0).contiguous()
                if self.normalize_embeddings:
                    all_p_reps = torch.nn.functional.normalize(all_p_reps, dim=-1)
                return all_p_reps.contiguous()
            else:
                last_hidden_state = self.doc_encoder(**features, return_dict=True).last_hidden_state
                all_p_reps = self._sentence_embedding(last_hidden_state, features['attention_mask'], sentence_pooling_method="last_token")
                if self.normalize_embeddings:
                    all_p_reps = torch.nn.functional.normalize(all_p_reps, dim=-1)
                return all_p_reps.contiguous()
        else:
            all_p_reps = []
            for sub_features in features:
                last_hidden_state = self.doc_encoder(**sub_features, return_dict=True).last_hidden_state
                p_reps = self._sentence_embedding(last_hidden_state, sub_features['attention_mask'], sentence_pooling_method="last_token")
                all_p_reps.append(p_reps)
            all_p_reps = torch.cat(all_p_reps, 0).contiguous()
            if self.normalize_embeddings:
                all_p_reps = torch.nn.functional.normalize(all_p_reps, dim=-1)
            return all_p_reps.contiguous()

    def _sentence_embedding(self, last_hidden_state, attention_mask, sentence_pooling_method=None):
        """Use the pooling method to get the sentence embedding.

        Args:
            last_hidden_state (torch.Tensor): The model output's last hidden state.
            attention_mask (torch.Tensor): Mask out padding tokens during pooling.
            sentence_pooling_method (str, optional): Pooling method to get sentence embedding. Defaults to ``None``.

        Raises:
            NotImplementedError: Specified pooling method not implemented.

        Returns:
            torch.Tensor: The sentence embeddings.
        """
        if sentence_pooling_method is None:
            sentence_pooling_method = self.sentence_pooling_method
        if sentence_pooling_method == "cls":
            return last_hidden_state[:, 0]
        elif sentence_pooling_method == "mean":
            s = torch.sum(
                last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1
            )
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d
        elif sentence_pooling_method == "last_token":
            left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
            if left_padding:
                return last_hidden_state[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_state.shape[0]
                return last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device),
                    sequence_lengths,
                ]
        else:
            raise NotImplementedError(f"pooling method {sentence_pooling_method} not implemented")

    def compute_score(self, q_reps, p_reps):
        """Computes the scores between query and passage representations.

        Args:
            q_reps (torch.Tensor): Query representations.
            p_reps (torch.Tensor): Passage representations.

        Returns:
            torch.Tensor: The computed scores, adjusted by temperature.
        """
        # if self.normalize_embeddings:
        #     q_reps = torch.nn.functional.normalize(q_reps, dim=-1)
        #     p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        scores = self._compute_similarity(q_reps, p_reps) / self.temperature
        scores = scores.view(q_reps.size(0), -1)
        return scores

    def _compute_similarity(self, q_reps, p_reps):
        """Computes the similarity between query and passage representations using inner product.

        Args:
            q_reps (torch.Tensor): Query representations.
            p_reps (torch.Tensor): Passage representations.

        Returns:
            torch.Tensor: The computed similarity matrix.
        """
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def compute_loss(self, scores, target):
        """Compute the loss using cross entropy.

        Args:
            scores (torch.Tensor): Computed score.
            target (torch.Tensor): The target value.

        Returns:
            torch.Tensor: The computed cross entropy loss.
        """
        return self.cross_entropy(scores, target)

    def gradient_checkpointing_enable(self, **kwargs):
        """
        Activates gradient checkpointing for the current model.
        """
        # self.model.gradient_checkpointing_enable(**kwargs)
        self.query_encoder.gradient_checkpointing_enable(**kwargs)
        self.doc_encoder.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self, **kwargs):
        """
        Enables the gradients for the input embeddings.
        """
        # self.model.enable_input_require_grads(**kwargs)
        self.query_encoder.enable_input_require_grads(**kwargs)
        self.doc_encoder.enable_input_require_grads(**kwargs)

    def _compute_cross_device_neg_loss(self, q_reps, p_reps, teacher_targets=None, compute_score_func=None, **kwargs):
        """
        Compute loss when using both in-batch negatives and cross-device negatives
        """
        group_size = p_reps.size(0) // q_reps.size(0)

        cross_q_reps = self._dist_gather_tensor(q_reps) # (world_size * batch_size, dim)
        cross_p_reps = self._dist_gather_tensor(p_reps) # (world_size * batch_size * group_size, dim)

        if self.use_mrl:
            loss = torch.tensor(0.0, device=q_reps.device)
            for num_dim in self.mrl_dims:
                cross_q_reps_mrl, cross_p_reps_mrl = cross_q_reps[..., :num_dim], cross_p_reps[..., :num_dim]
                cross_scores_mrl = self.compute_score(cross_q_reps_mrl, cross_p_reps_mrl)
                
                cross_idxs_mrl = torch.arange(cross_q_reps_mrl.size(0), device=cross_q_reps_mrl.device, dtype=torch.long)
                cross_targets_mrl = cross_idxs_mrl * group_size
                loss_mrl = self.compute_loss(cross_scores_mrl, cross_targets_mrl)
                loss+=loss_mrl
            
            loss = loss / len(self.mrl_dims)
            # cross_scores = self.compute_score(cross_q_reps, cross_p_reps)   # (world_size * batch_size, world_size * batch_size * group_size)
            # cross_idxs = torch.arange(cross_q_reps.size(0), device=cross_q_reps.device, dtype=torch.long)
            # cross_targets = cross_idxs * group_size # (world_size * batch_size)
            # loss = self.compute_loss(cross_scores, cross_targets)
        else:
            cross_scores = self.compute_score(cross_q_reps, cross_p_reps[..., :768])   # (world_size * batch_size, world_size * batch_size * group_size)
            cross_idxs = torch.arange(cross_q_reps.size(0), device=cross_q_reps.device, dtype=torch.long)
            cross_targets = cross_idxs * group_size # (world_size * batch_size)
            loss = self.compute_loss(cross_scores, cross_targets)      

        return loss


    def _compute_mse_loss(self, student_reps, teacher_reps, **kwargs):
        teacher_reps = teacher_reps[..., :768]
        if student_reps.shape != teacher_reps.shape:
            raise ValueError(f"embedding dimension doesn't align"
                            f"student: {student_reps.shape}, teacher: {teacher_reps.shape}")
        squared_l2_diff = torch.sum((student_reps - teacher_reps)**2, dim=1)

        mse_loss = torch.mean(squared_l2_diff)
        
        return mse_loss

    def _compute_cossim_loss(self, student_reps, teacher_reps, **kwargs):
        teacher_reps=teacher_reps[..., :768]
        target = torch.ones(student_reps.size(0), device=student_reps.device)
        loss = F.cosine_embedding_loss(student_reps, teacher_reps, target)
        return loss

    def forward(
        self, 
        queries: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None, 
        passages: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None,
        queries_doc_encoder: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None,
        passages_query_encoder: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None,
        teacher_scores: Union[None, List[float]] = None,
        no_in_batch_neg_flag: bool = False,
    ):
        """The computation performed at every call.

        Args:
            queries (Union[Dict[str, Tensor], List[Dict[str, Tensor]]], optional): Input queries. Defaults to ``None``.
            passages (Union[Dict[str, Tensor], List[Dict[str, Tensor]]], optional): Input passages. Defaults to ``None``.
            teacher_scores (Union[None, List[float]], optional): Teacher scores for distillation. Defaults to ``None``.
            no_in_batch_neg_flag (bool, optional): If True, use no in-batch negatives and no cross-device negatives. Defaults to ``False``.

        Returns:
            EmbedderOutput: Output of the forward call of model.
        """
        q_reps_q = self.encode_queries(queries) # (batch_size, dim)
        p_reps_d = self.encode_corpus(passages) # (batch_size * group_size, dim_p)

        if self.kd_loss_type=='contrastive':
            loss  = self._compute_cross_device_neg_loss(q_reps_q, p_reps_d)
        elif self.kd_loss_type=='mse':
            q_reps_d = self.encode_corpus(queries_doc_encoder)
            p_reps_q= self.encode_queries(passages_query_encoder)
            teacher_reps = torch.cat([q_reps_d, p_reps_d], dim=0).detach()
            student_reps = torch.cat([q_reps_q, p_reps_q], dim=0)
            loss = self._compute_mse_loss(student_reps, teacher_reps)
        elif self.kd_loss_type=='cossim':
            q_reps_d = self.encode_corpus(queries_doc_encoder)
            p_reps_q= self.encode_queries(passages_query_encoder)
            teacher_reps = torch.cat([q_reps_d, p_reps_d], dim=0).detach()
            student_reps = torch.cat([q_reps_q, p_reps_q], dim=0)
            loss = self._compute_cossim_loss(student_reps, teacher_reps)
        elif self.kd_loss_type=='contrastive_and_cossim':
            loss_contrastive = self._compute_cross_device_neg_loss(q_reps_q, p_reps_d)

            q_reps_d = self.encode_corpus(queries_doc_encoder)
            p_reps_q = self.encode_queries(passages_query_encoder)
            teacher_reps = torch.cat([q_reps_d, p_reps_d], dim=0).detach()
            student_reps = torch.cat([q_reps_q, p_reps_q], dim=0)
            loss_cossim = self._compute_cossim_loss(student_reps, teacher_reps)
            if dist.get_rank()==0:
                logging.info(f"\Contrastive: {loss_contrastive.item():.4f}, Cossim: {loss_cossim.item():.4f}")

            loss = loss_contrastive + loss_cossim

        elif self.kd_loss_type=='contrastive_and_mse':
            loss_contrastive = self._compute_cross_device_neg_loss(q_reps_q, p_reps_d)

            q_reps_d = self.encode_corpus(queries_doc_encoder)
            p_reps_q = self.encode_queries(passages_query_encoder)
            teacher_reps = torch.cat([q_reps_d, p_reps_d], dim=0).detach()
            student_reps = torch.cat([q_reps_q, p_reps_q], dim=0)
            loss_mse = self._compute_mse_loss(student_reps, teacher_reps)
            if dist.get_rank()==0:
                logging.info(f"\Contrastive: {loss_contrastive.item():.4f}, Mse: {loss_mse.item():.4f}")

            loss = loss_contrastive + loss_mse

        return EmbedderOutput(
            loss=loss,
        )

    def save(self, output_dir: str):
        """Save the model to the directory.

        Args:
            output_dir (str): Directory for saving the model.
        """
        def _trans_state_dict(state_dict):
            state_dict = type(state_dict)(
                {k: v.clone().cpu()
                 for k,
                 v in state_dict.items()})
            return state_dict
        
        query_encoder_save_dir = os.path.join(output_dir, "query_encoder")
        doc_encoder_save_dir = os.path.join(output_dir, "doc_encoder")
        
        self.query_encoder.save_pretrained(query_encoder_save_dir, state_dict=_trans_state_dict(self.query_encoder.state_dict()))
        
        self.doc_encoder.save_pretrained(doc_encoder_save_dir, state_dict=_trans_state_dict(self.doc_encoder.state_dict()))
        
