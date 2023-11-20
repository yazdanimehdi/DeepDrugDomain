from functools import partial
from typing import Optional, Sequence

from deepdrugdomain.layers.utils import LayerFactory, ActivationFactory

from ..factory import ModelFactory
import torch
from torch import nn
import torch.nn.functional as F
from deepdrugdomain.utils.weight_init import trunc_normal_
from torch.autograd import Variable

# Weird attention layer, but I guess it works!


@LayerFactory.register('drugvqa_seq_attention')
class DrugVQASequentialAttention(nn.Module):
    def __init__(self, num_heads: int, dim_input: int, dim_hidden: int, activation_fn: str = "tanh", ) -> None:
        super().__init__()
        self.first_linear = nn.Linear(dim_input, dim_hidden)
        self.activation_fn = ActivationFactory.create(activation_fn)
        self.second_linear = nn.Linear(dim_hidden, num_heads)
        self.num_heads = num_heads

    def forward(self, x):
        att = self.activation_fn(self.first_linear(x))
        att = self.linear_second(att)
        att = torch.softmax(att, axis=1)
        att = att.transpose(1, 2)
        embed = att @ x
        embed = torch.sum(embed, 1) / self.num_heads
        return embed


@ModelFactory.register('drugvqa')
class DrugVQA(nn.Module):
    def __init__(self,
                 attention_layers_smile: str,
                 attention_layers_smile_kwargs: dict,
                 attention_layers_seq: str,
                 attention_layers_seq_kwargs: dict,
                 contact_map_in_channels: int,
                 contact_map_out_channels: int,
                 contact_map_kernel_size: int,
                 contact_map_stride: int,
                 contact_map_activation_fn: str,
                 contact_map_normalization_layer: str,
                 vocab_size_smiles: int,
                 vocab_size_seq: int,
                 embedding_dim: int,
                 resnet_block1: str,
                 resnet_block1_kwargs: dict,
                 resnet_block1_layers: int,
                 resnet_block2: str,
                 resnet_block2_kwargs: dict,
                 resnet_block2_layers: int,
                 lstm_hid_dim: int,
                 lstm_layers: int,
                 lstm_bidirectional: bool,
                 lstm_dropout: float,
                 num_batches: int,
                 head_dropout_rate: float,
                 head_activation_fn: Optional[str],
                 head_normalization: Optional[str],
                 head_dims: Sequence[int]):
        super().__init__()

        self.lstm_hid_dim = lstm_hid_dim

        # rnn
        self.embeddings = nn.Embedding(vocab_size_smiles, embedding_dim)
        self.seq_embed = nn.Embedding(vocab_size_seq, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, self.lstm_hid_dim, lstm_layers,
                                  batch_first=True, bidirectional=lstm_bidirectional, dropout=lstm_dropout)
        self.smile_attention = LayerFactory.create(
            attention_layers_smile, **attention_layers_smile_kwargs)

        # cnn
        self.conv = nn.Conv2d(contact_map_in_channels, contact_map_out_channels,
                              kernel_size=contact_map_kernel_size, stride=contact_map_stride)
        self.bn = LayerFactory.create(
            contact_map_normalization_layer, self.in_channels)
        self.contact_map_act = ActivationFactory.create(
            contact_map_activation_fn)

        self.res_block1 = nn.Sequential(*[LayerFactory.create(
            resnet_block1, inplanes=contact_map_out_channels, planes=contact_map_out_channels, **resnet_block1_kwargs) for _ in range(resnet_block1_layers)])

        self.res_block2 = nn.Sequential(*[LayerFactory.create(
            resnet_block2, inplanes=contact_map_out_channels, planes=contact_map_out_channels, **resnet_block2_kwargs) for _ in range(resnet_block2_layers)])

        h_0 = Variable(torch.zeros(2 * lstm_layers,
                       num_batches, self.embedding_dim))
        c_0 = Variable(torch.zeros(2 * lstm_layers,
                       num_batches, self.embedding_dim))
        self.lstm_hidden_state = (h_0, c_0)

        self.seq_attention = LayerFactory.create(
            attention_layers_seq, **attention_layers_seq_kwargs)
        # Prediction layer
        self.head = nn.ModuleList()
        neuron_list = [self.lstm_hid_dim * lstm_layers +
                       attention_layers_seq_kwargs["dim_hidden"]] + list(head_dims)
        for item in range(len(neuron_list) - 1):
            self.head.append(nn.Dropout(head_dropout_rate))
            self.head.append(LayerFactory.create(head_normalization)
                             if head_normalization else nn.Identity())
            self.head.append(
                nn.Linear(neuron_list[item], neuron_list[item + 1]))
            self.head.append(ActivationFactory.create(
                head_activation_fn) if head_activation_fn else nn.Identity())

        trunc_normal_(self.latent_query, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def forward(self, x1, x2):
        smile_embed = self.embeddings(x1)
        outputs, self.lstm_hidden_state = self.lstm(
            smile_embed, self.lstm_hidden_state)
        avg_sentence_embed = self.smile_attention(outputs)

        pic = self.conv(x2)
        pic = self.bn(pic)
        pic = self.contact_map_act(pic)
        pic = self.res_block1(pic)
        pic = self.res_block2(pic)
        pic_emb = torch.mean(pic, 2)
        pic_emb = pic_emb.permute(0, 2, 1)
        avg_seq_embed = self.seq_attention(pic_emb)

        out = torch.cat([avg_sentence_embed, avg_seq_embed], dim=1)

        return self.head(out)
