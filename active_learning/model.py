import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel

from active_learning.loss import LabelSmoothLoss


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier_3 = nn.Linear(config.hidden_size, num_labels)

        self.loss = LabelSmoothLoss(num_labels)

    def forward(self, input_ids, y_ids, token_type_ids=None, attention_mask=None, **kwargs):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = self.classifier_2(logits)
        logits = self.classifier_3(logits)

        loss = self.loss(logits, y_ids)

        return loss, torch.softmax(logits, dim=-1)






