from transformers import HfArgumentParser

from FlagEmbedding.abc.finetune.embedder import (
    AbsEmbedderDataArguments as AsymmetricEmbedderDataArguments,
)

from runner import AsymmetricEmbedderRunner
from arguments import (
    AsymmetricEmbedderModelArguments, AsymmetricEmbedderTrainingArguments
)


def main():
    parser = HfArgumentParser((
        AsymmetricEmbedderModelArguments,
        AsymmetricEmbedderDataArguments,
        AsymmetricEmbedderTrainingArguments
    ))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: AsymmetricEmbedderModelArguments
    data_args: AsymmetricEmbedderDataArguments
    training_args: AsymmetricEmbedderTrainingArguments

    runner = AsymmetricEmbedderRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )
    runner.run()


if __name__ == "__main__":
    main()
