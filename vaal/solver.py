import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class VAE(BertPreTrainedModel):
    def __init__(self, config, vocab_lm_size, args):
        super(VAE, self).__init__(config)
        # encoder
        self.bert = BertModel(config)
        self.enc_linear = nn.Linear(768, 1024)
        self.batch_norm = nn.BatchNorm1d(1024)

        self.fc_mu = nn.Linear(1024, args.z_dim)
        self.fc_logvar = nn.Linear(1024, args.z_dim)

        # decoder
        self.dec_linear_l1 = nn.Linear(args.z_dim, args.rec_max_length*768)
        self.dec_batch_norm1 = nn.BatchNorm1d(768)
        self.dec_linear_l2 = nn.Linear(768, 768)
        self.dec_batch_norm2 = nn.BatchNorm1d(768)
        self.dec_linear_l3 = nn.Linear(768, vocab_lm_size)

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, X_ids, att_mask):
        b, s = X_ids.size()

        # encode
        x_seq_enc, x_enc = self.bert(X_ids, attention_mask=att_mask)  # b,h
        x_enc = self.enc_linear(x_enc)
        x_enc = self.batch_norm(x_enc)
        z = torch.relu(x_enc)

        # latent space
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        z = self.reparameterize(mu, logvar)

        # decode
        x_dec = self.dec_linear_l1(z)  # [b,s*768]
        x_dec = x_dec.view((b,s,768)).contiguous()
        x_dec = self.dec_batch_norm1(x_dec.transpose(1,2)).transpose(1,2)
        x_dec = torch.relu(x_dec)
        x_dec = self.dec_linear_l2(x_dec)  # [b,s,768]
        x_dec = self.dec_batch_norm2(x_dec.transpose(1,2)).transpose(1,2)
        x_dec = torch.relu(x_dec)

        logits = self.dec_linear_l3(x_dec)  # [b,s,2608]
        # logits = torch.softmax(logits, dim=-1)

        return logits, mu, logvar, z

    def reparameterize(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        if mu.is_cuda:
            stds, epsilon = stds.cuda(), epsilon.cuda()
        latents = epsilon * stds + mu
        return latents


class Discriminator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, z_dim//2),
            nn.ReLU(True),
            nn.Linear(z_dim//2, z_dim//2),
            nn.ReLU(True),
            nn.Linear(z_dim//2, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        logits = self.net(z)
        return logits.squeeze(-1)

