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


class TextCNN(nn.Module):
    def __init__(self, embeddings, num_labels):
        super(TextCNN, self).__init__()
        self.char_embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=150, kernel_size=(k, 150), padding=(k - 1, 0)) for k in [2, 3, 4]]
        )

        self.fc1 = nn.Linear(450, 150)

        self.fc2 = nn.Linear(150, num_labels)

    def forward(self, input: torch.Tensor):
        x_emb = self.char_embedding(input)

        xs = [torch.relu(conv(x_emb.unsqueeze(1))).squeeze(3) for conv in self.convs]
        xm = [torch.max_pool1d(x, kernel_size=x.size(2)).squeeze(2) for x in xs]
        xc = torch.cat(xm, 1)  # [b,450]

        output = self.fc1(xc)
        logits = self.fc2(output)

        logits = torch.log_softmax(logits, dim=-1)

        return logits



