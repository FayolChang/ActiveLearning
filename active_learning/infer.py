import json
from pathlib import Path

from dataclasses import dataclass, field
from transformers import AutoConfig, HfArgumentParser

from active_learning.data_generator import DataGenerator
from active_learning.model import BertForSequenceClassification
import torch
import numpy as np

from active_learning.training_args import TrainingArguments
from configuration.config import data_dir, intent_labels, roberta_model_path

from utils.vocab import load_vocab

###############################################
# args
###############################################
@dataclass
class ModelArguments:
    model_path_or_name: str = field(default=str(roberta_model_path))
    # model_path_or_name: str = field(default=str(Path(data_dir)/'checkpoints'/'checkpoint-6000'))


@dataclass
class DataTrainingArguments:
    max_seq_length: int = field(default=200)


parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()


###############################################
# model
###############################################
num_labels = len(intent_labels)
config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_args.model_path_or_name, num_labels=num_labels)
model = BertForSequenceClassification(config, num_labels)
model.load_state_dict(torch.load(Path(data_dir) / 'cls_model.pt', map_location='cuda' if torch.cuda.is_available() else "cpu"))


model.to(training_args.device)


###############################################
# data process
###############################################
d_80 = [(_['text'], _['label']) for _ in json.load((Path(data_dir) / 'data_40_per_u.json').open())]
vocabulary = load_vocab(vocab_file=(Path(roberta_model_path) / 'vocab.txt'))

d_80_loader = DataGenerator(d_80, training_args, data_args, vocabulary, intent_labels)


###############################################
# train
###############################################
d_80_score = []
for k, batch in enumerate(d_80_loader):

    # if k > 0: break  # debug

    raw_text = batch[-1]
    batch = [_.to(training_args.device) for _ in batch[:-1]]
    X_ids, Y_ids, Mask = batch
    with torch.no_grad():
        _, logits = model(X_ids, Y_ids, Mask)

    for logit, y_id, t in zip(logits, Y_ids, raw_text):
        logit = logit.detach().cpu().numpy()
        true_label = y_id.detach().cpu().numpy()

        pred_label = np.argmax(logit)
        score = max(logit)

        d_80_score.append({
            'text': t,
            'true_label': intent_labels[y_id],
            'pred_label': intent_labels[pred_label],
            'score': f'{score:.4f}',
        })
json.dump(d_80_score, (Path(data_dir)/'d_40.json').open('w'), ensure_ascii=False, indent=2)
print('Done')

