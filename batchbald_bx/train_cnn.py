import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, \
    get_linear_schedule_with_warmup, HfArgumentParser

from active_learning.loss import LabelSmoothLoss
from active_learning.training_args import TrainingArguments
from batchbald_bx.data_generator_2 import DataGeneratorW2V
from batchbald_bx.model_2 import TextCNN
from configuration.config import data_dir, intent_labels, common_data_path, logger, bert_model_path


def train_main(train_loader, vocabulary, id2embeddings):
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

    global_step = 0


    ###############################################
    # distant debug
    ###############################################
    if training_args.server_ip and training_args.server_port:
        import ptvsd
        print('Waiting for debugger attach')
        ptvsd.enable_attach(address='')



    ###############################################
    # model
    ###############################################
    id2embeddings = torch.tensor(id2embeddings, dtype=torch.float).to(training_args.device)

    num_labels = len(intent_labels)
    model = TextCNN(id2embeddings, num_labels)


    ###############################################
    # data process
    ###############################################
    dev = [(_['text'], _['label']) for _ in json.load((Path(common_data_path)/'intent_data' / 'dev_data.json').open())]

    dev_loader = DataGeneratorW2V(dev, training_args.batch_size, data_args, vocabulary, intent_labels)

    ###############################################
    # optimizer
    ###############################################
    optimizer = torch.optim.Adam([p for n, p in list(model.named_parameters())], lr=5e-5)


    ###############################################
    # train
    ###############################################
    model.to(training_args.device)
    logger.info(f'gpu num: {training_args.n_gpu}')
    if training_args.n_gpu > 1:
        model = nn.DataParallel(model)

    loss_func = LabelSmoothLoss(num_labels)
    best_acc = 0
    best_epoch = 0
    model.zero_grad()
    for e in range(training_args.epoch_num):
    # for e in range(1):  # debug
        model.train()
        t_loss = 0
        logging_loss = 0
        for step, batch in enumerate(train_loader):

            # if step > 0: break  # debug

            raw_text = batch[-1]
            batch = [_.to(training_args.device) for _ in batch[:-1]]
            X_ids, Y_ids = batch
            if step < 1: logger.info(f'batch_size: {X_ids.size()[0]}')
            logits = model(X_ids, 1).squeeze(1)
            loss = loss_func(Y_ids, logits)

            if training_args.n_gpu > 1:
                loss = loss.mean()
            loss.backward()
            t_loss += loss.item()

            if training_args.gradient_accumulation_steps > 1:
                loss = loss / training_args.gradient_accumulation_steps

            if ((step + 1) % training_args.gradient_accumulation_steps == 0
                    or (train_loader.steps <= training_args.gradient_accumulation_steps) and step + 1 == train_loader.steps):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=training_args.max_gradient_norm)
                optimizer.step()
                model.zero_grad()

                global_step += 1

                if global_step % training_args.logging_steps == 0:
                    logger.info(f'epoch: {e} - batch: {step}/{train_loader.steps} - loss: {t_loss / (step + 1): 6f}')

        model.eval()
        dev_acc = 0
        eval_loss = 0
        err = []
        cat = defaultdict(lambda: 1e-10)
        for k, batch in enumerate(dev_loader):

            # if k > 0: break  # debug

            raw_text = batch[-1]
            batch = [_.to(training_args.device) for _ in batch[:-1]]
            X_ids, Y_ids = batch
            with torch.no_grad():
                logits = model(X_ids, 1).squeeze(1)
                loss = loss_func(Y_ids, logits)

                if training_args.n_gpu > 1:
                    loss = loss.mean()
                eval_loss += loss.item()

            for logit, y_id, t in zip(logits, Y_ids, raw_text):
                logit = logit.detach().cpu().numpy()
                true_label = y_id.detach().cpu().numpy()

                pred_label = np.argmax(logit)

                # metric 1
                if true_label == pred_label:
                    dev_acc += 1
                else:
                    score = max(logit)
                    err.append({
                        'text': t,
                        'pred': intent_labels[pred_label],
                        'true': intent_labels[true_label],
                        'score': f'{score: .4f}'

                    })
                # metric 2
                cat[f'{intent_labels[true_label]}_A'] += int(pred_label == true_label)
                cat[f'{intent_labels[true_label]}_B'] += 1
                cat[f'{intent_labels[pred_label]}_C'] += 1
        acc = dev_acc / (len(dev_loader)*training_args.batch_size)

        if acc > best_acc:
        # if acc >= best_acc:  # debug
            best_acc = acc
            best_epoch = e

            metric = {
                'epoch': best_epoch,
                'acc': best_acc
            }

            # save #
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), Path(data_dir) / f'cls_model.pt')

            # save #
            json.dump(err, (Path(data_dir) / 'err.json').open('w'), ensure_ascii=False, indent=4)

        logger.info(f'epoch: {e} - dev_acc: {acc:.5f} {dev_acc}/{len(dev_loader)*training_args.batch_size} - '
                    f'best_score: {best_acc:.5f} - best_epoch: {best_epoch} ')
        # for t in intent_labels:
        #     logger.info(f'cat: {t} - '
        #                 f'precision: {cat[t + "_A"] / cat[t + "_C"]:.5f} - '
        #                 f'recall: {cat[t + "_A"] / cat[t + "_B"]:.5f} - '
        #                 f'f1: {2 * cat[t + "_A"] / (cat[t + "_B"] + cat[t + "_C"]):.5f}')


    return metric, model_to_save





