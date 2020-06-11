import json
import logging
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
    get_linear_schedule_with_warmup, AutoConfig, HfArgumentParser

from active_learning.loss import LabelSmoothLoss
from active_learning.training_args import TrainingArguments
from batchbald_bx.data_generator_2 import DataGenerator
from batchbald_bx.model_2 import BertForSequenceClassification
from configuration.config import data_dir, roberta_model_path, intent_labels, common_data_path, logger, bert_model_path
from utils.vocab import load_vocab


def train_main(train_loader):
    """

    :param train_data:
    :return:
    """

    # in_file = Path(data_dir) / f'labeled_{p}.json' if isinstance(p, int) else Path(common_data_path) /'intent_data'/ p

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
    num_labels = len(intent_labels)
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_args.model_path_or_name, num_labels=num_labels)
    model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_args.model_path_or_name, config=config, num_labels=num_labels)


    ###############################################
    # data process
    ###############################################
    # train = [(_['text'], _['label']) for _ in json.load(in_file.open())]
    dev = [(_['text'], _['label']) for _ in json.load((Path(common_data_path)/'intent_data' / 'dev_data.json').open())]

    vocabulary = load_vocab()
    # vocabulary = load_vocab(vocab_file=(Path(roberta_model_path) / 'vocab.txt'))

    # train_loader = DataGenerator(train, training_args, data_args, vocabulary, intent_labels, shuffle=True)
    dev_loader = DataGenerator(dev, training_args.batch_size, data_args, vocabulary, intent_labels)

    ###############################################
    # optimizer
    ###############################################
    def get_optimizer(num_training_steps):
        no_decay = ['bias', 'LayerNorm.weight']
        optimize_group_params = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': training_args.weight_decay

            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = AdamW(optimize_group_params, lr=training_args.learning_rate, weight_decay=training_args.weight_decay)

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=training_args.warmup_steps,
                                                    num_training_steps=num_training_steps)

        return optimizer, scheduler


    optimizer, scheduler = get_optimizer(num_training_steps=len(train_loader) * training_args.epoch_num / training_args.batch_size)


    ###############################################
    # continue training from checkpoints
    ###############################################
    if (
            'checkpoint' in model_args.model_path_or_name
            and os.path.isfile(os.path.join(model_args.model_path_or_name, 'optimizer.pt'))
            and os.path.isfile(os.path.join(model_args.model_path_or_name, 'scheduler.pt'))
    ):
        optimizer.load_state_dict(torch.load(os.path.join(model_args.model_path_or_name, "optimizer.pt"), map_location='cuda' if torch.cuda.is_available() else 'cpu'))
        scheduler.load_state_dict(torch.load(os.path.join(model_args.model_path_or_name, "scheduler.pt"), map_location='cuda' if torch.cuda.is_available() else 'cpu'))

    epoch_trained = 0
    step_trained_cur_epoch = 0
    if 'checkpoint' in model_args.model_path_or_name:
        global_step = int(str(Path(model_args.model_path_or_name)).split('-')[-1].split('/')[0])
        epoch_trained = global_step // (train_loader.steps // training_args.gradient_accumulation_steps)
        step_trained_cur_epoch = global_step % (train_loader.steps // training_args.gradient_accumulation_steps)

        logger.info(' Continuing Training from checkpoint, will skip to saved global_step')
        logger.info(f' Continuing Training from epoch {epoch_trained}')
        logger.info(f' Continuing Training from global step {global_step}')
        logger.info(f' Will skip the first {step_trained_cur_epoch} steps in the first epoch')


    ###############################################
    # tensorboard
    ###############################################
    tb_writer = SummaryWriter(log_dir=Path(data_dir) / 'logs')
    def tb_log(logs):
        for k_, v_ in logs.items():
            tb_writer.add_scalar(k_, v_, global_step)
    tb_writer.add_text('args', training_args.to_json_string())
    tb_writer.add_hparams(training_args.to_sanitized_dict(), metric_dict={})


    ###############################################
    # save
    ###############################################
    def save_model(output_dir, model):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f'Saving model checkpoint to {output_dir}')

        model_to_save = model.module if hasattr(model, 'module') else model

        model_to_save.config.architectures = [model_to_save.__class__.__name__]  # architectures是什么

        output_model_file = os.path.join(output_dir, 'pytorch.bin')
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info(f'Model weights saved in {output_model_file}')

        output_config_file = os.path.join(output_dir, 'config.json')
        model_to_save.config.to_json_file(output_config_file)
        logger.info(f'Configuration saved in {output_config_file}')

        torch.save(training_args, os.path.join(output_dir, 'training_args.bin'))


    def sorted_checkpoints(checkpoint_prefix="checkpoint", use_mtime=False):
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(training_args.output_dir).glob(f"{checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted


    def rotate_checkpoints(use_mtime=False) -> None:
        if training_args.save_total_limit is None or training_args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = sorted_checkpoints(use_mtime=use_mtime)
        if len(checkpoints_sorted) <= training_args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - training_args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
            shutil.rmtree(checkpoint)


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
    for e in range(epoch_trained, training_args.epoch_num):
    # for e in range(1):  # debug
        model.train()
        t_loss = 0
        logging_loss = 0
        for step, batch in enumerate(train_loader):

            # if step > 0: break  # debug

            if step_trained_cur_epoch > 0:
                step_trained_cur_epoch -= 1
                continue

            raw_text = batch[-1]
            batch = [_.to(training_args.device) for _ in batch[:-1]]
            X_ids, Y_ids, Mask = batch
            if step < 5: logger.info(f'batch_size: {X_ids.size()}')
            logits = model(X_ids, 1, attention_mask=Mask).squeeze(1)
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
                scheduler.step()
                model.zero_grad()

                global_step += 1
                epoch = e + (step + 1) / train_loader.steps

                if global_step % training_args.logging_steps == 0:
                    train_logs = {
                        'loss': (t_loss - logging_loss) / training_args.logging_steps,
                        'learning_rate': scheduler.get_lr()[0],
                        'epoch': epoch
                    }

                    logging_loss = t_loss
                    tb_log(train_logs)

                    logger.info(f'epoch: {e} - batch: {step}/{train_loader.steps} - loss: {t_loss / (step + 1): 6f}')

                # if global_step % training_args.saving_steps == 0:
                #     output_dir = os.path.join(training_args.output_dir, f'checkpoint-{global_step}')
                #
                #     save_model(output_dir, model)
                #     rotate_checkpoints()
                #
                #     torch.save(optimizer.state_dict(), Path(output_dir)/'optimizer.pt')
                #     torch.save(scheduler.state_dict(), Path(output_dir)/'scheduler.pt')
                #     logger.info(f'Saving optimizer and scheduler states to {output_dir}')

        model.eval()
        dev_acc = 0
        eval_loss = 0
        err = []
        cat = defaultdict(lambda: 1e-10)
        for k, batch in enumerate(dev_loader):

            # if k > 0: break  # debug

            raw_text = batch[-1]
            batch = [_.to(training_args.device) for _ in batch[:-1]]
            X_ids, Y_ids, Mask = batch
            with torch.no_grad():
                logits = model(X_ids, 1, attention_mask=Mask).squeeze(1)
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
        acc = dev_acc / len(dev_loader)

        eval_logs = {
            'eval_acc': acc,
            'eval_loss': eval_loss / dev_loader.steps,
        }
        tb_log(eval_logs)

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

        logger.info(f'epoch: {e} - dev_acc: {acc:.5f} {dev_acc}/{len(dev_loader)} - best_score: {best_acc:.5f} - best_epoch: {best_epoch} ')
        # for t in intent_labels:
        #     logger.info(f'cat: {t} - '
        #                 f'precision: {cat[t + "_A"] / cat[t + "_C"]:.5f} - '
        #                 f'recall: {cat[t + "_A"] / cat[t + "_B"]:.5f} - '
        #                 f'f1: {2 * cat[t + "_A"] / (cat[t + "_B"] + cat[t + "_C"]):.5f}')

    tb_writer.close()

    return metric, model_to_save





