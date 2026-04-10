from dataclasses import dataclass
from typing import Union
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from FlagEmbedding.abc.finetune.embedder.AbsDataset import AbsEmbedderCollator, AbsEmbedderSameDatasetCollator


@dataclass
class AsymmetricEmbedderCollator(AbsEmbedderCollator):
    query_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None
    doc_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None
    
    def __call__(self, features):
        queries = [f[0] for f in features]
        passages = [f[1] for f in features]
        teacher_scores = [f[2] for f in features]
        if teacher_scores[0] is None:
            teacher_scores = None
        elif isinstance(teacher_scores[0], list):
            teacher_scores = sum(teacher_scores, [])

        if isinstance(queries[0], list):
            queries = sum(queries, [])
        if isinstance(passages[0], list):
            passages = sum(passages, [])

        queries_inputs = self.query_tokenizer(
            queries,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors=None
        )
        passages_inputs = self.doc_tokenizer(
            passages,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors=None
        )

        if self.sub_batch_size is None or self.sub_batch_size <= 0:
            q_collated = self.query_tokenizer.pad(
                queries_inputs,
                padding=self.padding,
                max_length=self.query_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors
            )
            d_collated = self.doc_tokenizer.pad(
                passages_inputs,
                padding=self.padding,
                max_length=self.passage_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors
            )
        else:
            batch_size = self.sub_batch_size

            q_collated = []
            for i in range(0, len(queries_inputs['attention_mask']), batch_size):
                start = i
                end = min(len(queries_inputs['attention_mask']), i + batch_size)
                sub_features = {}
                for k, v in queries_inputs.items():
                    sub_features[k] = v[start:end]
                q_collated.append(self.query_tokenizer.pad(
                    sub_features,
                    padding=self.padding,
                    max_length=self.passage_max_len,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors
                ))

            d_collated = []
            for i in range(0, len(passages_inputs['attention_mask']), batch_size):
                start = i
                end = min(len(passages_inputs['attention_mask']), i + batch_size)
                sub_features = {}

                for k, v in passages_inputs.items():
                    sub_features[k] = v[start:end]
                d_collated.append(self.doc_tokenizer.pad(
                    sub_features,
                    padding=self.padding,
                    max_length=self.passage_max_len,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors
                ))
        return {
            "queries": q_collated,
            "passages": d_collated,
            "teacher_scores": teacher_scores,
            "no_in_batch_neg_flag": False
        }


@dataclass
class AsymmetricEmbedderSameDatasetCollator(AbsEmbedderSameDatasetCollator):
    query_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None
    doc_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None
    
    def __call__(self, features):
        queries = features[0][0]
        passages = features[0][1]
        teacher_scores = features[0][2]
        no_in_batch_neg_flag = features[0][3]

        queries_inputs = self.query_tokenizer(
            queries,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors=None
        )
        passages_inputs = self.doc_tokenizer(
            passages,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors=None
        )

        if self.sub_batch_size is None or self.sub_batch_size <= 0:
            q_collated = self.query_tokenizer.pad(
                queries_inputs,
                padding=self.padding,
                max_length=self.query_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )

            d_collated = self.doc_tokenizer.pad(
                passages_inputs,
                padding=self.padding,
                max_length=self.passage_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
        else:
            batch_size = self.sub_batch_size

            q_collated = []
            for i in range(0, len(queries_inputs['attention_mask']), batch_size):
                start = i
                end = min(len(queries_inputs['attention_mask']), i + batch_size)
                sub_features = {}
                for k, v in queries_inputs.items():
                    sub_features[k] = v[start:end]
                q_collated.append(self.query_tokenizer.pad(
                    sub_features,
                    padding=self.padding,
                    max_length=self.query_max_len,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors,
                ))

            d_collated = []
            for i in range(0, len(passages_inputs['attention_mask']), batch_size):
                start = i
                end = min(len(passages_inputs['attention_mask']), i + batch_size)
                sub_features = {}

                for k, v in passages_inputs.items():
                    sub_features[k] = v[start:end]
                d_collated.append(self.doc_tokenizer.pad(
                    sub_features,
                    padding=self.padding,
                    max_length=self.passage_max_len,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors,
                ))

        if isinstance(teacher_scores, list) and len(teacher_scores) == 0:
            teacher_scores = None

        return {
            "queries": q_collated,
            "passages": d_collated,
            "teacher_scores": teacher_scores,
            "no_in_batch_neg_flag": no_in_batch_neg_flag
        }

@dataclass
class AsymmetricEmbedderSameDatasetCollator_distill(AbsEmbedderSameDatasetCollator):
    query_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None
    doc_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None
    
    def __call__(self, features):
        queries = features[0][0]
        passages = features[0][1]
        teacher_scores = features[0][2]
        no_in_batch_neg_flag = features[0][3]

        queries_inputs = self.query_tokenizer(
            queries,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors=None
        )
        queries_inputs_doc_encoder = self.doc_tokenizer(
            queries,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors=None
        )
        passages_inputs = self.doc_tokenizer(
            passages,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors=None
        )
        passages_inputs_query_encoder = self.query_tokenizer(
            passages,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors=None
        )

        if self.sub_batch_size is None or self.sub_batch_size <= 0:
            q_collated = self.query_tokenizer.pad(
                queries_inputs,
                padding=self.padding,
                max_length=self.query_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            q_collated_doc_encoder = self.doc_tokenizer.pad(
                queries_inputs_doc_encoder,
                padding=self.padding,
                max_length=self.query_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )

            d_collated = self.doc_tokenizer.pad(
                passages_inputs,
                padding=self.padding,
                max_length=self.passage_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            d_collated_query_encoder = self.query_tokenizer.pad(
                passages_inputs_query_encoder,
                padding=self.padding,
                max_length=self.passage_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,      
            )
        else:
            batch_size = self.sub_batch_size

            q_collated = []
            for i in range(0, len(queries_inputs['attention_mask']), batch_size):
                start = i
                end = min(len(queries_inputs['attention_mask']), i + batch_size)
                sub_features = {}
                for k, v in queries_inputs.items():
                    sub_features[k] = v[start:end]
                q_collated.append(self.query_tokenizer.pad(
                    sub_features,
                    padding=self.padding,
                    max_length=self.query_max_len,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors,
                ))

            q_collated_doc_encoder = []
            for i in range(0, len(queries_inputs_doc_encoder['attention_mask']), batch_size):
                start = i
                end = min(len(queries_inputs_doc_encoder['attention_mask']), i + batch_size)
                sub_features = {}
                for k, v in queries_inputs_doc_encoder.items():
                    sub_features[k] = v[start:end]
                q_collated_doc_encoder.append(self.doc_tokenizer.pad(
                    sub_features,
                    padding=self.padding,
                    max_length=self.query_max_len,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors,
                ))

            d_collated = []
            for i in range(0, len(passages_inputs['attention_mask']), batch_size):
                start = i
                end = min(len(passages_inputs['attention_mask']), i + batch_size)
                sub_features = {}

                for k, v in passages_inputs.items():
                    sub_features[k] = v[start:end]
                d_collated.append(self.doc_tokenizer.pad(
                    sub_features,
                    padding=self.padding,
                    max_length=self.passage_max_len,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors,
                ))


            d_collated_query_encoder = []
            for i in range(0, len(passages_inputs_query_encoder['attention_mask']), batch_size):
                start = i
                end = min(len(passages_inputs_query_encoder['attention_mask']), i + batch_size)
                sub_features = {}

                for k, v in passages_inputs_query_encoder.items():
                    sub_features[k] = v[start:end]
                d_collated_query_encoder.append(self.query_tokenizer.pad(
                    sub_features,
                    padding=self.padding,
                    max_length=self.passage_max_len,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors,
                ))

        if isinstance(teacher_scores, list) and len(teacher_scores) == 0:
            teacher_scores = None

        return {
            "queries": q_collated,
            "passages": d_collated,
            "queries_doc_encoder":q_collated_doc_encoder,
            "passages_query_encoder":d_collated_query_encoder,
            "teacher_scores": teacher_scores,
            "no_in_batch_neg_flag": no_in_batch_neg_flag
        }