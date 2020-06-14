import torch
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


class textcnn(consistent_mc_dropout.BayesianModule):
    def __init__(self, embeddings, num_labels):
        super(textcnn, self).__init__()
        self.char_embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.conv = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=150, kernel_size=(k, 150), padding=(k - 1, 0)) for k in [2, 3, 4]]
        )
        self.conv_drop = nn.ModuleList(
            [consistent_mc_dropout.ConsistentMCDropout2d() for _ in range(3)]
        )

        self.fc1 = nn.Linear(450, 150)
        self.fc1_drop = consistent_mc_dropout.ConsistentMCDropout()

        self.fc2 = nn.Linear(150, num_labels)

    def mc_forward_impl(self, input: torch.Tensor):
        x_emb = self.char_embedding(input)

        xs = [F.relu(conv_drop(conv(x_emb.unsqueeze(1)))).squeeze(3) for conv, conv_drop in zip(self.conv, self.conv_drop)]
        xm = [F.max_pool1d(x, kernel_size=x.size(2)).squeeze(2) for x in xs]
        xc = torch.cat(xm, 1)  # [b,450]

        output = self.fc1(xc)
        output = self.fc1_drop(output)
        logits = self.fc2(output)

        logits = F.log_softmax(logits, dim=-1)

        return logits




