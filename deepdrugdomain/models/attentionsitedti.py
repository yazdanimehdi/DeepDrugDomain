import torch
from dgl.nn.pytorch import MaxPooling
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from layers import Attention
from typing import Callable, Optional, Sequence


class AttentionSiteDTI(nn.Module):

    def __init__(self,
                 protein_graph_conv_layer: Callable = "TAGConv",
                 protein_graph_conv_bias: bool = True,
                 protein_graph_conv_activation: Optional[Callable] = None,
                 # protein_graph_conv_activation
                 ligand_graph_conv_layer: str = "TAGConv",
                 ligand_graph_conv_bias: bool = True,
                 ligand_graph_conv_activation: Optional[Callable] = None,

                 protein_input_size: int = 74,
                 ligand_input_size: int = 74,
                 protein_graph_conv_dims: Sequence[int, ...] = (50, 45),
                 ligand_graph_conv_dims: Sequence[int, ...] = (50, 45, 45),
                 sequence_length: int = 130,
                 embedding_dim: int = 45,
                 ligand_graph_pooling: Callable = MaxPooling,
                 protein_graph_pooling: Callable = MaxPooling,
                 use_lstm_layer: bool = False,
                 use_bilstm: bool = False,
                 lstm_dims: Optional[Sequence[int, ...]] = None,
                 head_dims: Sequence[int, ...] = (2000, 1000, 500, 1),
                 attention_layer: Callable = Attention,
                 protein_conv_dropout_rate: float = 0.3,
                 ligand_conv_dropout_rate: float = 0.3,
                 head_dropout_rate: float = 0.3,
                 ):
        super(AttentionSiteDTI, self).__init__()
        self.embedding_dim = embedding_dim
        GraphConv()
        TAGConv()
        GATConv()
        GINConv()
        self.protein_graph_conv = nn.ModuleList([protein_graph_conv_layer()])
        self.protein_graph_conv.append(TAGConv(74, 50, 4))
        self.protein_graph_conv.append(TAGConv(50, 45, 4))
        self.protein_graph_conv.append(TAGConv(45, 45, 4))

        self.ligand_graph_conv = nn.ModuleList()

        self.ligand_graph_conv.append(TAGConv(74, 50, 4))
        self.ligand_graph_conv.append(TAGConv(50, 45, 4))
        self.ligand_graph_conv.append(TAGConv(45, 45, 4))
        self.ligand_graph_conv.append(TAGConv(45, 45, 4))

        pool_ligand = MaxPooling()
        pool_protein = MaxPooling()

        self.pooling_ligand = nn.Linear(31, 1)
        self.pooling_protein = nn.Linear(31, 1)


        self.dropout = 0.3
        self.dropout_prot = nn.Dropout()
        self.bilstm = nn.LSTM(20, 20, num_layers=1, bidirectional=True, dropout=self.dropout)

        self.linear_module = nn.ModuleList()
        neuron_list = [self.size*130, 2000, 1000, 500, 1]
        for item in range(len(neuron_list) - 2):
            self.linear_module.append(nn.Linear(neuron_list[item], neuron_list[item+1]))

        self.fc_out = nn.Linear(neuron_list[-2], neuron_list[-1])

        self.attention = Attention(self.size, num_heads=1, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.)

    def forward(self, g, training):
        feature_protein = g[0].ndata['h']
        feature_smile = g[1].ndata['h']
        dropout = nn.Dropout(self.dropout)
        for module in self.protein_graph_conv:
            feature_protein = F.relu(module(g[0], feature_protein))
            feature_protein = F.normalize(feature_protein)
            feature_protein = dropout(feature_protein)

        for module in self.ligand_graph_conv:
            feature_smile = F.relu(module(g[1], feature_smile))
            feature_smile = F.normalize(feature_smile)
            eature_smile = dropout(feature_smile)

        # pool_ligand = GlobalAttentionPooling(self.pooling_ligand)
        # pool_protein = GlobalAttentionPooling(self.pooling_protein)


        protein_rep = pool_protein(g[0], feature_protein).view(-1, self.size)
        ligand_rep = pool_ligand(g[1], feature_smile).view(-1, self.size)
        print(ligand_rep.shape)
        #sequence = []
        #for item in protein_rep:
        #    sequence.append(item.view(1, 31))
        #    sequence.append(ligand_rep)

        sequence = torch.cat((ligand_rep, protein_rep), dim=0).view(1, -1, self.size)
        mask = torch.eye(130, dtype=torch.uint8).view(1, 130, 130).cuda()
        mask[0, sequence.size()[1]:130, :] = 0
        mask[0, :, sequence.size()[1]:130] = 0
        mask[0, :, sequence.size()[1] - 1] = 1
        mask[0, sequence.size()[1] - 1, :] = 1
        mask[0, sequence.size()[1] - 1, sequence.size()[1] - 1] = 0
        sequence = F.pad(input=sequence, pad=(0, 0, 0, 130 - sequence.size()[1]), mode='constant', value=0)
        # sequence = sequence.permute(1, 0, 2)

        h_0 = Variable(torch.zeros(6, 1, 31).cuda())
        c_0 = Variable(torch.zeros(6, 1, 31).cuda())

        # output, _ = self.bilstm(sequence, (h_0, c_0))

        # output = output.permute(1, 0,  2)
        # output = self.cnn1(output)
        # output = self.cnn2(output)
        out, att = self.attention(sequence, mask=mask, return_attention=True)
        #attn_weight_matrix = self.attention_net(output)
        #out = torch.bmm(attn_weight_matrix, output)
        # out = F.dropout(out, self.dropout)
        # out = F.max_pool2d(self.cnn(out), 2)
        # out = F.max_pool2d(self.cnn2(out), 2)
        # out = out.flatten()
        out = out.view(-1, out.size()[1]*out.size()[2])
        # out = F.dropout(out, self.dropout)
        for layer in self.linear_module:
            out = F.relu(layer(out))
            out = F.dropout(out, self.dropout, training=training)
        out = torch.sigmoid(self.fc_out(out))
        return out
