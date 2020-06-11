import json
from pathlib import Path

import torch
from dataclasses import dataclass, field
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import HfArgumentParser

from active_learning.training_args import TrainingArguments
from batchbald_bx.data_generator_2 import DataGenerator
from batchbald_bx.train_2 import train_main
from batchbald_redux import active_learning, batchbald
from batchbald_redux.repeated_mnist import get_targets
from configuration.config import common_data_path, intent_labels, bert_model_path
from utils.vocab import load_vocab


###############################################
# args
###############################################
@dataclass
class ModelArguments:
    model_path_or_name: str = field(default=str(bert_model_path))
    # model_path_or_name: str = field(default=str(roberta_model_path))
    # model_path_or_name: str = field(default=str(Path(data_dir)/'checkpoints'/'checkpoint-6000'))


@dataclass
class DataTrainingArguments:
    max_seq_length: int = field(default=200)


parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

train_file = Path(common_data_path) / 'intent_data' / 'train_data.json'
num_classes = len(intent_labels)
num_initial_samples = 500
max_training_samples = 5000
num_inference_samples = 100
acquisition_batch_size = 5
num_samples = 100000
pool_batch_size = 128


class IntentDataset(Dataset):
    def __init__(self, file_name):
        self.data, self.targets = zip(*[(_['text'], _['label']) for _ in json.load(file_name.open())])
        self._label2id()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def _label2id(self):
        self.targets = [intent_labels.index(t) for t in self.targets]


intent_train_set = IntentDataset(train_file)


# get initial samples
initial_samples = active_learning.get_balanced_sample_indices(
    intent_train_set.targets,
    num_classes=num_classes,
    n_per_digit=num_initial_samples//num_classes
)


# build active learning data
active_learning_data = active_learning.ActiveLearningData(intent_train_set)

active_learning_data.acquire(initial_samples)


# build train generator
vocabulary = load_vocab()
train_generator = DataGenerator(active_learning_data.training_dataset, training_args.batch_size, data_args, vocabulary, intent_labels, shuffle=True)
pool_generator = DataGenerator(active_learning_data.pool_dataset, pool_batch_size, data_args, vocabulary, intent_labels, shuffle=True)

use_cuda = torch.cuda.is_available()

# Run experiment
added_indices = []
pbar = tqdm(initial=len(active_learning_data.training_dataset),
            total=max_training_samples,
            desc="Training Set Size")


while True:
    metric, model = train_main(train_generator)

    if len(active_learning_data.training_dataset) >= max_training_samples:
        break

    # Acquire pool predictions
    N = len(active_learning_data.pool_dataset)
    logits_N_K_C = torch.empty((N, num_inference_samples, num_classes),
                               dtype=torch.double,
                               pin_memory=use_cuda)

    with torch.no_grad():
        model.eval()

        for i, batch in enumerate(
                tqdm(pool_generator,
                     desc="Evaluating Acquisition Set",
                     leave=False)):

            batch = [_.to(training_args.device) for _ in batch[:-1]]
            X_ids, Y_ids, Mask = batch

            lower = i * pool_batch_size
            upper = min(lower + pool_batch_size, N)
            logits_N_K_C[lower:upper].copy_(model(
                X_ids, num_inference_samples, attention_mask=Mask).double(),
                                            non_blocking=True)

    with torch.no_grad():
        candidate_batch = batchbald.get_batchbald_batch(logits_N_K_C.exp_(),
                                                        acquisition_batch_size,
                                                        num_samples,
                                                        dtype=torch.double,
                                                        device=training_args.device)

    targets = get_targets(active_learning_data.pool_dataset)
    dataset_indices = active_learning_data.get_dataset_indices(
        candidate_batch.indices)

    print("Dataset indices: ", dataset_indices)
    print("Scores: ", candidate_batch.scores)
    print("Labels: ", targets[candidate_batch.indices])
    print('Metric: ')
    print(metric)

    active_learning_data.acquire(candidate_batch.indices)
    added_indices.append(dataset_indices)
    pbar.update(len(dataset_indices))

