import collections
import json
from pathlib import Path

import torch
import torch.nn as nn
from dataclasses import field, dataclass
from torch.utils import data
from transformers import HfArgumentParser, AutoConfig

from batchbald_redux.repeated_mnist import get_targets
from configuration.config import common_data_path, intent_labels, data_dir, bert_model_path, logger
from universal_data_generators import data_generator
from universal_models.model import TextCNN
from utils.vocab import load_vocab_w2v
from vaal import trainer, sampler, load_simple_vocab
from vaal.al import ActiveLearningData, get_balanced_sample_indices
#######################################
# args
#######################################
from vaal.solver import VAE, Discriminator
from vaal.training_args import TrainingArguments

num_init_samples = 500
pool_batch_size = 128
budget = 500


@dataclass
class DataTrainingArguments:
    max_seq_length: int = field(default=200)


parser = HfArgumentParser((DataTrainingArguments, TrainingArguments))
data_args, training_args = parser.parse_args_into_dataclasses()

logger.info(f'n_gpu: {training_args.n_gpu}')

#######################################
# data prepare
#######################################
class IntentDataset(data.Dataset):
    def __init__(self, file_name):
        self.data, self.targets = zip(
            *[(_['text'], _['label']) for _ in json.load(file_name.open()) if _['label'] != '负样本'])
        self._label2id()

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def __len__(self):
        return len(self.data)

    def _label2id(self):
        self.targets = [intent_labels.index(t) for t in self.targets]


num_classes = len(intent_labels)

f_name = Path(common_data_path)/'intent_data/train_data.json'
intent_dataset = IntentDataset(f_name)
active_learning_data = ActiveLearningData(intent_dataset)

if (Path(data_dir)/'initial_samples_indices.json').exists():
    initial_samples_indices = json.load((Path(data_dir)/'initial_samples_indices.json').open())
else:
    initial_samples_indices = get_balanced_sample_indices(intent_dataset.targets, intent_labels,
                                                          n_per_class=num_init_samples // num_classes)
    json.dump(initial_samples_indices, (Path(data_dir)/'initial_samples_indices.json').open('w'), ensure_ascii=False)

active_learning_data.acquire(initial_samples_indices)

vocab_wv, id2embeddings = load_vocab_w2v()
vocab_lm = load_simple_vocab.load_vocab(remove_sign=True)
train_generator = data_generator.DataGeneratorW2V_VAE(active_learning_data.train_data,
                                                      training_args.batch_size,
                                                      data_args,
                                                      vocab_wv,
                                                      vocab_lm,
                                                      intent_labels,
                                                      shuffle=True)
pool_generator = data_generator.DataGeneratorW2V_VAE(active_learning_data.pool_data,
                                                     pool_batch_size,
                                                     data_args,
                                                     vocab_wv,
                                                     vocab_lm,
                                                     intent_labels)

dev_data = [(_['text'], _['label']) for _ in json.load((Path(common_data_path)/'intent_data/dev_data.json').open())]
dev_generator = data_generator.DataGeneratorW2V_VAE(dev_data,
                                                    training_args.batch_size,
                                                    data_args,
                                                    vocab_wv,
                                                    vocab_lm,
                                                    intent_labels)

#######################################
# model construction
#######################################
task_model = TextCNN(torch.tensor(id2embeddings, dtype=torch.float, device=training_args.device), num_labels=num_classes)
config = AutoConfig.from_pretrained(pretrained_model_name_or_path=str(bert_model_path))
vae = VAE.from_pretrained(pretrained_model_name_or_path=str(bert_model_path),
                          config=config,
                          z_dim=256,
                          vocab_size=len(vocab_lm))
discriminator = Discriminator(z_dim=256)

task_model.to(training_args.device)
vae.to(training_args.device)
discriminator.to(training_args.device)

if training_args.n_gpu > 0:
    task_model = nn.DataParallel(task_model)
    vae = nn.DataParallel(vae)
    discriminator = nn.DataParallel(discriminator)


while True:

    if len(active_learning_data.train_data) > 5000:
        break

    # train
    metric = trainer.train(train_generator, dev_generator, pool_generator, task_model, vae, discriminator, training_args)
    logger.info(f'train size: {len(active_learning_data.train_data)}')
    logger.info(f'dev acc: {metric}')

    # sample
    # candidate_indices = sampler.sample(vae, discriminator, pool_generator, budget, training_args)
    # data_indices = active_learning_data.get_pool_indices(candidate_indices)
    # targets = get_targets(active_learning_data.pool_data)
    # candidate_targets = targets[data_indices].detach().cpu().numpy()
    # candidate_targets_names = [intent_labels[_] for _ in candidate_targets]
    #
    # logger.info(f'acquired label distribution: ')
    # logger.info(collections.Counter(candidate_targets_names).most_common())
    #
    # active_learning_data.acquire(candidate_indices)

















