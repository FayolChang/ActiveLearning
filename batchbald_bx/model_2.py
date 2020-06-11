import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel

from batchbald_redux import consistent_mc_dropout

"""
todo: add mc dropout

"""


class BertForSequenceClassification(BertPreTrainedModel, consistent_mc_dropout.BayesianModule):
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.bert_drop = consistent_mc_dropout.ConsistentMCDropout()

        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc1_drop = consistent_mc_dropout.ConsistentMCDropout()

        self.fc2 = nn.Linear(config.hidden_size, num_labels)

    def mc_forward_impl(self, input_ids, **kwargs):
        _, pooled_output = self.bert(input_ids, kwargs["attention_mask"])
        pooled_output = self.bert_drop(pooled_output)
        output = self.fc1(pooled_output)
        output = self.fc1_drop(output)
        logits = self.fc2(output)

        logits = F.log_softmax(logits, dim=-1)

        return logits






