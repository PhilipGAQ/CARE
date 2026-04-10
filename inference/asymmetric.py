from tqdm import tqdm, trange
from typing import cast, Any, List, Union, Optional, Dict, Literal
import logging
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig
import queue
import multiprocessing as mp
from multiprocessing import Queue
from FlagEmbedding.abc.inference import AbsEmbedder
from transformers.configuration_utils import PretrainedConfig
from FlagEmbedding import FlagAutoModel

logger = logging.getLogger(__name__)

INSTRUCTION_FORMAT='<instruct>{}\n<query>{}'
INSTRUCTION_DICT={
    'CMedQAv2-reranking':"Based on a Chinese medical question, evaluate and rank the medical information that provide answers to the question.",
    'CMedQAv1-reranking':"Based on a Chinese medical question, evaluate and rank the medical information that provide answers to the question.",
    'retrieval':"Given a Chinese medical question, retrieve medical documents that answer the question.",
    'medteb_reranking':"Based on a Chinese medical question, evaluate and rank the medical information that provide answers to the question.",
    'medteb_sts_v1':"Retrieve semantically similar text.",
    'medteb_clustering':"Identify the category of disease based on the medical text."
}

def get_instruct(prompt_type:str, task_name: str, sentence: str):
    """Combine the instruction and sentence along with the instruction format.

    Args:
        instruction_format (str): Format for instruction.
        instruction (str): The text of instruction.
        sentence (str): The sentence to concatenate with.

    Returns:
        str: The complete sentence with instruction
    """
    if task_name=='medteb_clustering' or task_name=='medteb_sts':
        assert task_name in INSTRUCTION_DICT.keys(), f'{task_name} not in instruction dict!!!'    
        instruction = INSTRUCTION_DICT[task_name]
        return INSTRUCTION_FORMAT.format(instruction, sentence)

    if prompt_type=='query':
        assert task_name in INSTRUCTION_DICT.keys(), f'{task_name} not in instruction dict!!!'    
        instruction = INSTRUCTION_DICT[task_name]
        return INSTRUCTION_FORMAT.format(instruction, sentence)
    elif prompt_type=='passage':
        return sentence
    else:
        raise ValueError(f'{prompt_type}, {task_name}, {sentence} does not match any instructions')

class CARE(AbsEmbedder):
    DEFAULT_POOLING_METHOD = "cls"

    def __init__(
        self,
        model_name_or_path: str=None,
        normalize_embeddings: bool = True,
        use_fp16: bool = True,
        use_bf16: bool = False,
        query_instruction_for_retrieval: Optional[str] = None,
        query_instruction_format: str = "{}{}", # specify the format of query_instruction_for_retrieval
        devices: Optional[Union[str, List[str]]] = None, # specify devices, such as "cuda:0" or ["cuda:0", "cuda:1"]
        # Additional parameters for BaseEmbedder
        pooling_method: str = "cls",
        trust_remote_code: bool = False,
        cache_dir: Optional[str] = None,
        # inference
        batch_size: int = 256,
        query_max_length: int = 512,
        passage_max_length: int = 512,
        convert_to_numpy: bool = True,

        # added args
        model_name_or_path_query: str=None,
        model_name_or_path_doc: str=None,
        query_batch_size:int=16,
        passage_batch_size:int=16,
        save_name:str=None,

        **kwargs: Any,
    ):
        super().__init__(
            model_name_or_path,
            normalize_embeddings=normalize_embeddings,
            use_fp16=use_fp16,
            query_instruction_for_retrieval=query_instruction_for_retrieval,
            query_instruction_format=query_instruction_format,
            devices=devices,
            batch_size=batch_size,
            query_max_length=query_max_length,
            passage_max_length=passage_max_length,
            convert_to_numpy=convert_to_numpy,
            **kwargs
        )
        self.trust_remote_code = trust_remote_code
        self.cache_dir = cache_dir
        self.model_name_or_path_doc = model_name_or_path_doc
        self.model_name_or_path_query = model_name_or_path_query

        self.pooling_method = pooling_method
        self.additional_special_tokens = None
        self.token=None
        self.query_batch_size = query_batch_size
        self.passage_batch_size = passage_batch_size
        tokenizer, model = self.load_tokenizer_and_model()
        self.query_tokenizer =tokenizer['query_tokenizer']
        self.doc_tokenizer = tokenizer['doc_tokenizer']
        self.query_encoder = model['query_encoder']
        self.doc_encoder = model['doc_encoder']
        self.use_bf16 = use_bf16

    def load_tokenizer_and_model(self):
        """Load the tokenizer and model.

        Returns:
            Tuple[PreTrainedTokenizer, AbsEmbedderModel]: Tokenizer and model instances.
        """
        query_tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path_query,
            trust_remote_code=self.trust_remote_code
        )
        query_encoder = AutoModel.from_pretrained(
            self.model_name_or_path_query,
            trust_remote_code=self.trust_remote_code
        )
        
        doc_tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path_doc,
            token=self.token,
            use_fast=False,
            add_eos_token=True,
            trust_remote_code=self.trust_remote_code,
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
        if self.additional_special_tokens is not None:
            special_tokens_dict = {'additional_special_tokens': self.additional_special_tokens}
            add_num = doc_tokenizer.add_special_tokens(special_tokens_dict)
            if add_num > 0:
                resize = True
                logger.info(f"Add {add_num} special tokens to the tokenizer. Special tokens: {self.additional_special_tokens}")
            else:
                logger.warning(f"Special tokens {self.additional_special_tokens} already exists in the tokenizer.")
        # doc_encoder = get_model(self, self.training_args.output_dir, resize, len(doc_tokenizer))
        # load doc encoder

        num_labels = 1
        query_config = AutoConfig.from_pretrained(
            self.model_name_or_path_query,
            num_labels=num_labels,
            token=self.token,
            trust_remote_code=self.trust_remote_code,
        )
        logger.info('Query Encoder Config: %s', query_config)
        
        doc_config = AutoConfig.from_pretrained(
            self.model_name_or_path_doc,
            num_labels=num_labels,
            token=self.token,
            trust_remote_code=self.trust_remote_code,
        )
        doc_encoder = AutoModel.from_pretrained(
            self.model_name_or_path_doc,
            # torch_dtype=torch.bfloat16,
            use_flash_attention_2=False,
            token=self.token,
            from_tf=False,
            config=doc_config,
            trust_remote_code=self.trust_remote_code,
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
        return tokenizer, base_model

    def encode(
        self,
        sentences: Union[List[str], str],
        task_name = None,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = 512,
        convert_to_numpy: Optional[bool] = None,
        instruction: Optional[str] = None,
        instruction_format: Optional[str] = None,
        prompt_type=None,
        **kwargs: Any
    ):
        """encode the input sentences with the embedding model.

        Args:
            sentences (Union[List[str], str]): Input sentences to encode.
            batch_size (Optional[int], optional): Number of sentences for each iter. Defaults to :data:`None`.
            max_length (Optional[int], optional): Maximum length of tokens. Defaults to :data:`None`.
            convert_to_numpy (Optional[bool], optional): If True, the output embedding will be a Numpy array. Otherwise, it will 
                be a Torch Tensor. Defaults to :data:`None`.
            instruction (Optional[str], optional): The text of instruction. Defaults to :data:`None`.
            instruction_format (Optional[str], optional): Format for instruction. Defaults to :data:`None`.

        Returns:
            Union[torch.Tensor, np.ndarray]: return the embedding vectors in a numpy array or tensor.
        """
        if batch_size is None: batch_size = self.batch_size
        if max_length is None: max_length = self.passage_max_length
        if convert_to_numpy is None: convert_to_numpy = self.convert_to_numpy

        if isinstance(sentences, str):
            sentences = get_instruct(prompt_type, task_name, sentences)
        else:
            sentences = [get_instruct(prompt_type, task_name, sentence) for sentence in
                            sentences]

        return self.encode_single_device(
            sentences,
            batch_size=batch_size,
            max_length=max_length,
            convert_to_numpy=convert_to_numpy,
            device=self.target_devices[0],
            prompt_type=prompt_type,
            **kwargs
        )

    @torch.no_grad()
    def encode_single_device(
        self,
        sentences: Union[List[str], str],
        batch_size: int = 256,
        max_length: int = 512,
        convert_to_numpy: bool = True,
        device: Optional[str] = None,
        prompt_type: str =None,
        show_progress_bar=False,
        convert_to_tensor=False,
        **kwargs: Any
    ):
        """Encode input sentences by a single device.

        Args:
            sentences (Union[List[str], str]): Input sentences to encode.
            batch_size (int, optional): Number of sentences for each iter. Defaults to :data:`256`.
            max_length (int, optional): Maximum length of tokens. Defaults to :data:`512`.
            convert_to_numpy (bool, optional): If True, the output embedding will be a Numpy array. Otherwise, it will 
                be a Torch Tensor. Defaults to :data:`True`.
            device (Optional[str], optional): Device to use for encoding. Defaults to None.

        Returns:
            Union[torch.Tensor, np.ndarray]: return the embedding vectors in a numpy array or tensor.
        """
        # encoder_type = kwargs.get('encoder_type', None)
        assert prompt_type in ['query', 'passage']
        if device is None:
            device = self.target_devices[0]
        if device == "cpu": self.use_fp16 = False
        if prompt_type == 'query': 
            if self.use_fp16:
                self.query_encoder = self.query_encoder.half()
            elif self.use_bf16:
                self.query_encoder = self.query_encoder.bfloat16()
            self.query_encoder.to(device)
            self.query_encoder.eval()
            tokenizer = self.query_tokenizer

        elif prompt_type == 'passage':
            if self.use_fp16:
                self.doc_encoder = self.doc_encoder.half()
            elif self.use_bf16:
                self.doc_encoder = self.doc_encoder.bfloat16()
            self.doc_encoder.to(device)
            self.doc_encoder.eval()
            tokenizer = self.doc_tokenizer

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        # tokenize without padding to get the correct length
        
        all_inputs = []
        for start_index in trange(0, len(sentences), batch_size, desc='pre tokenize',
                                  disable=len(sentences) < 256):
            sentences_batch = sentences[start_index:start_index + batch_size]
            inputs_batch = tokenizer(
                sentences_batch,
                truncation=True,
                max_length=max_length,
                **kwargs
            )
            inputs_batch = [{
                k: inputs_batch[k][i] for k in inputs_batch.keys()
            } for i in range(len(sentences_batch))]
            all_inputs.extend(inputs_batch)

        # sort by length for less padding
        length_sorted_idx = np.argsort([-len(x['input_ids']) for x in all_inputs])
        all_inputs_sorted = [all_inputs[i] for i in length_sorted_idx]

        # encode
        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Inference Embeddings",
                                disable=len(sentences) < 256):
            inputs_batch = all_inputs_sorted[start_index:start_index + batch_size]
            inputs_batch = tokenizer.pad(
                inputs_batch,
                padding=True,
                return_tensors='pt',
                **kwargs
            ).to(device)

            if prompt_type == 'query':
                last_hidden_state = self.query_encoder(**inputs_batch, return_dict=True).last_hidden_state
                embeddings = self._sentence_embedding(last_hidden_state, inputs_batch['attention_mask'], sentence_pooling_method='cls')
            elif prompt_type == 'passage':
                last_hidden_state = self.doc_encoder(**inputs_batch, return_dict=True).last_hidden_state
                embeddings = self._sentence_embedding(last_hidden_state, inputs_batch['attention_mask'], sentence_pooling_method="last_token")
                # if self.normalize_embeddings:
                #     embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = cast(torch.Tensor, embeddings)

            if convert_to_numpy:
                embeddings = embeddings.float().cpu().numpy()
            all_embeddings.append(embeddings)

        if convert_to_numpy:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            all_embeddings = torch.cat(all_embeddings, dim=0)

        # adjust the order of embeddings
        all_embeddings = all_embeddings[np.argsort(length_sorted_idx)]

        # return the embeddings
        if input_was_string:
            return all_embeddings[0][..., :768]
        return all_embeddings[..., :768]

    def encode_queries(
        self,
        queries: Union[List[str], str],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        convert_to_numpy: Optional[bool] = None,
        **kwargs: Any
    ):
        if convert_to_numpy is None: convert_to_numpy = self.convert_to_numpy

        return self.encode(
            queries,
            batch_size=self.query_batch_size,
            max_length=self.query_max_length,
            convert_to_numpy=convert_to_numpy,
            prompt_type='query',
            **kwargs
        )

    def encode_corpus(
        self,
        corpus: Union[List[str], str],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        convert_to_numpy: Optional[bool] = None,
        **kwargs: Any
    ):

        if convert_to_numpy is None: convert_to_numpy = self.convert_to_numpy

        return self.encode(
            corpus,
            batch_size=self.passage_batch_size,
            max_length=self.passage_max_length,
            convert_to_numpy=convert_to_numpy,
            prompt_type='passage',
            **kwargs
        )[..., :768]

    def _sentence_embedding(self, last_hidden_state, attention_mask, sentence_pooling_method=None):
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
        

if __name__ == "__main__":
    model_name_or_path_query = ""
    model_name_or_path_doc = ""
    care = CARE(
        model_name_or_path_query=model_name_or_path_query,
        model_name_or_path_doc=model_name_or_path_doc,
        trust_remote_code=True,
        use_fp16=False,
        normalize_embeddings=True,
        query_batch_size=2,
        passage_batch_size=2,
    )
    queries = [
        "什么是高血压？"
    ]
    corpus = [
        "高血压是指动脉血压持续升高，通常指收缩压≥140mmHg和/或舒张压≥90mmHg。"
    ]
    query_embeddings = care.encode_queries(queries, task_name='retrieval')
    print("Query Embeddings:", query_embeddings, query_embeddings.shape)
    corpus_embeddings = care.encode_corpus(corpus, task_name='retrieval')
    print("Corpus Embeddings:", corpus_embeddings, corpus_embeddings.shape)
    
    scores = np.dot(query_embeddings, corpus_embeddings.T)
    print("Similarity Scores:", scores)