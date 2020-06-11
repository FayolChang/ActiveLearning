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

    def forward(self, input_ids, k, **kwargs):
        _, input_B = self.bert(input_ids, kwargs["attention_mask"])

        consistent_mc_dropout.BayesianModule.k = k
        mc_input_BK = consistent_mc_dropout.BayesianModule.mc_tensor(input_B, k)
        mc_output_BK = self.mc_forward_impl(mc_input_BK)
        mc_output_B_K = consistent_mc_dropout.BayesianModule.unflatten_tensor(mc_output_BK, k)
        return mc_output_B_K

    def mc_forward_impl(self, mc_output_BK):
        output = self.fc1(mc_output_BK)
        output = self.fc1_drop(output)
        logits = self.fc2(output)

        logits = F.log_softmax(logits, dim=-1)

        return logits






