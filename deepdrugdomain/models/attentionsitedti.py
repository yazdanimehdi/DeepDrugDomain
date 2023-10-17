"""
AttentionSiteDTI: Implementation of an Interpretable Graph-Based Model for Drug-Target Interaction Prediction

The AttentionSiteDTI model is inspired by the advancements in Natural Language Processing, specifically
in sentence-level relation classification. The model represents the drug and target as graphs, and through
a combination of graph convolutional layers, attention mechanisms, and LSTM layers, it produces interaction
predictions between them.

The unique combination of LSTM layers with graph-based representations allows this model to consider both
sequential and structural patterns in the drug-target interaction. This results in a more holistic and
informed prediction, emphasizing regions crucial for interaction.

The implemented architecture can be summarized as:
- Graph Convolutional Layers: Transform node features in both drug and target graphs.
- LSTM Layers (optional): To capture sequential patterns in the combined representation.
- Attention Layer: Provides weights to the interaction regions based on importance.
- Classification Layer: Final prediction of the drug-target interaction.

Authors of the original paper: [Author1, Author2, ...] (Please replace with actual authors)

Citation:
[Please provide the actual citation for the paper here.]

"""
import math

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from deepdrugdomain.layers import Attention, GraphLayerFactory
from typing import Callable, Optional, Sequence

from deepdrugdomain.utils.weight_init import trunc_normal_


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attn=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)
        if return_attn:
            return o, attention
        else:
            return o


class AttentionSiteDTI(nn.Module):
    """
        The AttentionSiteDTI model is designed for Drug-Target Interaction (DTI) prediction using attention mechanisms.
        It processes both protein and drug graphs and aggregates their representations.

        Attributes:
        - Protein graph convolution layers
        - Ligand graph convolution layers
        - Graph pooling layers
        - LSTM (optional) for sequence modeling
        - Attention layer
        - Output prediction layer

        Args:
        protein_graph_conv_layer : str
            Type of graph convolution layer for protein (default is "dgl_tag").
        ligand_graph_conv_layer : str
            Type of graph convolution layer for drug (default is "dgl_tag").
        ... [other arguments]

        Note: For more details on each argument, refer to the method definition below.
    """

    def __init__(self,
                 protein_graph_conv_layer: str = "dgl_tag",
                 ligand_graph_conv_layer: str = "dgl_tag",
                 protein_input_size: int = 74,
                 ligand_input_size: int = 74,
                 protein_graph_conv_dims: Sequence[int] = (50, 45),
                 ligand_graph_conv_dims: Sequence[int] = (50, 45, 45),
                 sequence_length: int = 150,
                 embedding_dim: int = 45,
                 ligand_graph_pooling: str = "dgl_maxpool",
                 protein_graph_pooling: str = "dgl_maxpool",
                 use_lstm_layer: bool = False,
                 use_bilstm: bool = False,
                 lstm_input: Optional[int] = None,
                 lstm_output: Optional[int] = None,
                 lstm_num_layers: Optional[int] = None,
                 lstm_dropout_rate: Optional[float] = None,
                 head_dims: Sequence[int] = (2000, 1000, 500, 1),
                 attention_layer: Callable = Attention,
                 attention_head: int = 1,
                 attention_dropout: float = 0.0,
                 qk_scale: Optional[float] = None,
                 proj_drop: float = 0.0,
                 attention_layer_bias: bool = True,
                 protein_conv_dropout_rate: float = 0.2,
                 ligand_conv_dropout_rate: float = 0.2,
                 head_dropout_rate: float = 0.1,
                 head_activation_fn=nn.ReLU,
                 **kwargs,
                 ) -> None:
        super(AttentionSiteDTI, self).__init__()
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        kwargs['k'] = 8
        # Initialize protein graph convolution layers
        p_dims = [protein_input_size] + list(protein_graph_conv_dims)
        self.protein_graph_conv = nn.ModuleList([
            GraphLayerFactory.create(protein_graph_conv_layer,
                                     p_dims[i],
                                     p_dims[i + 1],
                                     **kwargs) for i in range(len(p_dims) - 1)])

        # Initialize drug graph convolution layers

        l_dims = [ligand_input_size] + list(ligand_graph_conv_dims)
        self.ligand_graph_conv = nn.ModuleList([
            GraphLayerFactory.create(ligand_graph_conv_layer,
                                     l_dims[i],
                                     l_dims[i + 1],
                                     **kwargs) for i in range(len(l_dims) - 1)])

        # Graph pooling layers
        self.pool_ligand = GraphLayerFactory.create(ligand_graph_pooling, **kwargs)
        self.pool_protein = GraphLayerFactory.create(protein_graph_pooling, **kwargs)

        self.protein_conv_dropout = nn.Dropout(protein_conv_dropout_rate)
        self.ligand_conv_dropout = nn.Dropout(ligand_conv_dropout_rate)
        self.head_dropout = nn.Dropout(head_dropout_rate)

        # Graph pooling layers
        self.use_lstm = use_lstm_layer
        if use_lstm_layer:
            assert None not in [lstm_input, lstm_output, lstm_dropout_rate,
                                lstm_num_layers], "You need to set the LSTM parameters in the model"
            self.lstm = nn.LSTM(lstm_input, self.embedding_dim, num_layers=lstm_num_layers, bidirectional=use_bilstm,
                                dropout=lstm_dropout_rate)
            self.h_0 = Variable(torch.zeros(lstm_num_layers * 2, 1, self.embedding_dim).cuda())
            self.c_0 = Variable(torch.zeros(lstm_num_layers * 2, 1, self.embedding_dim).cuda())

        else:
            self.lstm = nn.Identity()

        assert self.embedding_dim % attention_head == 0, "The embedding dimension must be advisable by number of \
                                                          attention heads"

        # Attention layer
        self.attention = attention_layer(self.embedding_dim, num_heads=attention_head,
                                         qkv_bias=attention_layer_bias, qk_scale=qk_scale,
                                         attn_drop=attention_dropout, proj_drop=proj_drop)

        # Prediction layer
        self.fc = nn.ModuleList()
        neuron_list = [self.embedding_dim * sequence_length] + list(head_dims)
        for item in range(len(neuron_list) - 2):
            self.fc.append(nn.Linear(neuron_list[item], neuron_list[item + 1]))

        self.fc_out = nn.Linear(neuron_list[-2], neuron_list[-1])

        self.activation = head_activation_fn()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, g):
        """
                Forward pass of the model.

                Args:
                g : tuple of DGLGraph
                    Tuple containing the protein and drug graphs.

                Returns:
                out : torch.Tensor
                    Prediction scores for the drug-target interaction.
        """
        feature_protein = g[0].ndata['h']
        feature_smile = g[1].ndata['h']
        for module in self.protein_graph_conv:
            feature_protein = F.relu(module(g[0], feature_protein))
            feature_protein = F.normalize(feature_protein)
            feature_protein = self.protein_conv_dropout(feature_protein)

        for module in self.ligand_graph_conv:
            feature_smile = F.relu(module(g[1], feature_smile))
            feature_smile = F.normalize(feature_smile)
            feature_smile = self.ligand_conv_dropout(feature_smile)

        protein_rep = self.pool_protein(g[0], feature_protein).view(-1, self.embedding_dim)
        ligand_rep = self.pool_ligand(g[1], feature_smile).view(-1, self.embedding_dim)

        sequence = torch.cat((ligand_rep, protein_rep), dim=0).view(1, -1, self.embedding_dim)
        mask = torch.eye(self.sequence_length, dtype=torch.uint8).view(1, self.sequence_length,
                                                                       self.sequence_length).cuda()
        mask[0, sequence.size()[1]:self.sequence_length, :] = 0
        mask[0, :, sequence.size()[1]:self.sequence_length] = 0
        mask[0, :, sequence.size()[1] - 1] = 1
        mask[0, sequence.size()[1] - 1, :] = 1
        mask[0, sequence.size()[1] - 1, sequence.size()[1] - 1] = 0
        sequence = F.pad(input=sequence, pad=(0, 0, 0, self.sequence_length - sequence.size()[1]), mode='constant',
                         value=0)

        sequence = sequence.permute(1, 0, 2)

        if self.use_lstm:
            output, _ = self.lstm(sequence, (self.h_0, self.c_0))
        else:
            output = sequence

        output = output.permute(1, 0, 2)

        out, att = self.attention(output, mask=mask, return_attn=True)

        out = out.view(-1, out.size()[1] * out.size()[2])
        # out = out[:, 0, :]
        for layer in self.fc:
            out = F.relu(layer(out))
            out = self.head_dropout(out)
        out = torch.sigmoid(self.fc_out(out))
        return out
