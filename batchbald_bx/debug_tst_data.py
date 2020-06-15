import json
from pathlib import Path

from dataclasses import dataclass, field
from torch.utils.data import Dataset
from transformers import HfArgumentParser

from active_learning.training_args import TrainingArguments
from batchbald_bx import train_cnn, train_2, data_generator_2
from utils.vocab import load_vocab_w2v, load_vocab

from batchbald_redux import active_learning
from configuration.config import common_data_path, intent_labels, bert_model_path


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
num_initial_samples = 2000
max_training_samples = 5000
num_inference_samples = 100
acquisition_batch_size = 3
num_samples = 1000000
pool_batch_size = 128

model_type = 'cnn'


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
if model_type == 'bert':
    vocabulary = load_vocab()
    train_generator = data_generator_2.DataGenerator(active_learning_data.training_dataset, training_args.batch_size, data_args, vocabulary, intent_labels, shuffle=True)
    pool_generator = data_generator_2.DataGenerator(active_learning_data.pool_dataset, pool_batch_size, data_args, vocabulary, intent_labels)
    train_2.train_main(train_generator)

else:
    vocabulary, id2embeddings = load_vocab_w2v()
    train_generator = data_generator_2.DataGeneratorW2V(active_learning_data.training_dataset, training_args.batch_size, data_args, vocabulary, intent_labels, shuffle=True)
    pool_generator = data_generator_2.DataGeneratorW2V(active_learning_data.pool_dataset, pool_batch_size, data_args, vocabulary, intent_labels)
    train_cnn.train_main(train_generator, vocabulary, id2embeddings)


# use_cuda = torch.cuda.is_available()
#
# # Run experiment
# added_indices = []
# pbar = tqdm(initial=len(active_learning_data.training_dataset),
#             total=max_training_samples,
#             desc="Training Set Size")


# while True:
#     metric, model = train_main(train_generator)
#
#     if len(active_learning_data.training_dataset) >= max_training_samples:
#         break
#
#     # Acquire pool predictions
#     N = len(active_learning_data.pool_dataset)
#     logits_N_K_C = torch.empty((N, num_inference_samples, num_classes),
#                                dtype=torch.double,
#                                pin_memory=use_cuda)
#
#     with torch.no_grad():
#         model.eval()
#
#         for i, batch in enumerate(
#                 tqdm(pool_generator,
#                      desc="Evaluating Acquisition Set",
#                      leave=False)):
#
#             batch = [_.to(training_args.device) for _ in batch[:-1]]
#             X_ids, Y_ids = batch
#
#             lower = i * pool_batch_size
#             upper = min(lower + pool_batch_size, N)
#             logits_N_K_C[lower:upper].copy_(model(
#                 X_ids, num_inference_samples).double(),
#                                             non_blocking=True)
#
#     with torch.no_grad():
#         candidate_batch = batchbald.get_batchbald_batch(logits_N_K_C.exp_(),
#                                                         acquisition_batch_size,
#                                                         num_samples,
#                                                         dtype=torch.double,
#                                                         device=training_args.device)
#
#     targets = get_targets(active_learning_data.pool_dataset)
#     dataset_indices = active_learning_data.get_dataset_indices(
#         candidate_batch.indices)
#
#     logger.info(f"Dataset indices: {dataset_indices}")
#     logger.info(f"Scores: {candidate_batch.scores}")
#     logger.info(f"Labels: {targets[candidate_batch.indices]}")
#     logger.info(f"Labels name: {[intent_labels[idx] for idx in targets[candidate_batch.indices].detach().cpu().numpy()]}")
#     logger.info('Metric: ')
#     logger.info(metric)
#
#     active_learning_data.acquire(candidate_batch.indices)
#     added_indices.append(dataset_indices)
#     pbar.update(len(dataset_indices))

