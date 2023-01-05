# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This is a modified version of 'run_glue.py' from HuggingFace's library.
"""

import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, EvalPrediction
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from utils import TemporalProcessor, convert_examples_to_features


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir."}
    )
    train_data: str = field(
        metadata={"help": "The input train data."}
    )
    dev_data: str = field(
        metadata={"help": "The input dev data dir"}
    )
    test_data: str = field(
        metadata={"help": "The input tes data dir"}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)
    num_labels = 2

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = RobertaConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        # finetuning_task=data_args.task_name,
    )
    tokenizer = RobertaTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = RobertaForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    processor = TemporalProcessor()
    label_list = processor.get_labels()

    train_dataset = None
    eval_dataset = None
    test_dataset = None

    if training_args.do_train:
        train_examples = processor.get_train_examples(data_args.train_data)
        train_dataset = convert_examples_to_features(train_examples, label_list, data_args.max_seq_length, tokenizer)

    if training_args.do_eval:
        eval_examples = processor.get_dev_examples(data_args.dev_data)
        eval_dataset = convert_examples_to_features(eval_examples,label_list, data_args.max_seq_length,  tokenizer)

    if training_args.do_predict:
        test_examples = processor.get_test_examples(data_args.test_data)
        test_dataset = convert_examples_to_features(test_examples, label_list, data_args.max_seq_length, tokenizer)

    # This function was adapted from McTACO repo.
    def compute_mctaco_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        predictions = [processor.get_labels()[val] for val in preds]

        gold_data = []
        if eval_dataset:
            gold_data = [x.strip() for x in open(data_args.dev_data).readlines()]
        elif test_dataset:
            gold_data = [x.strip() for x in open(data_args.test_data).readlines()]

        result_map = {}
        prediction_count_map = {}
        prediction_map = {}
        gold_label_count_map = {}
        for i, line in enumerate(gold_data):
            key = " ".join(line.split("\t")[0:2])
            if key not in result_map:
                result_map[key] = []
                prediction_count_map[key] = 0.0
                gold_label_count_map[key] = 0.0
                prediction_map[key] = []
            prediction = predictions[i]
            prediction_map[key].append(prediction)
            label = line.split("\t")[3]
            if prediction == "yes":
                prediction_count_map[key] += 1.0
            if label == "yes":
                gold_label_count_map[key] += 1.0
            result_map[key].append(prediction == label)

        total = 0.0
        correct = 0.0
        f1 = 0.0
        for question in result_map:
            val = True
            total += 1.0
            cur_correct = 0.0
            for i, v in enumerate(result_map[question]):
                val = val and v
                if v and prediction_map[question][i] == "yes":
                    cur_correct += 1.0
            if val:
                correct += 1.0
            p = 1.0
            if prediction_count_map[question] > 0.0:
                p = cur_correct / prediction_count_map[question]
            r = 1.0
            if gold_label_count_map[question] > 0.0:
                r = cur_correct / gold_label_count_map[question]
            if p + r > 0.0:
                f1 += 2 * p * r / (p + r)

        em = correct / total
        f1 = f1 / total

        return {"EM": em, "F1": f1}


    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_mctaco_metrics,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_result = trainer.evaluate(eval_dataset=eval_dataset)

        output_eval_file = os.path.join(training_args.output_dir, f"eval_results.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in eval_result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

        eval_results.update(eval_result)

    # Prediction
    if training_args.do_predict:
        logging.info("*** Test ***")

        predictions = trainer.predict(test_dataset=test_dataset).predictions
        predictions = np.argmax(predictions, axis=1)

        output_test_file = os.path.join(
            training_args.output_dir, f"qa_prediction.txt"
        )
        if trainer.is_world_master():
            with open(output_test_file, "w") as writer:
                logger.info("***** Test results *****")
                for index, item in enumerate(predictions):
                    label = processor.get_labels()[item]
                    writer.write(label + '\n')

    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()