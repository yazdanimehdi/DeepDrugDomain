from typing import Any, Sequence
from torch import Tensor, nn
import torch
import math
from deepdrugdomain.layers.utils.layer_factory import LayerFactory
from deepdrugdomain.layers.utils import ActivationFactory
import torch.nn.functional as F
from ..factory import ModelFactory
from deepdrugdomain.layers import get_node_attr, change_node_attr


@LayerFactory.register("ammvf_position_wise_ff")
class PositionWiseFeedforward(nn.Module):
    def __init__(self, hid_dim: int, pf_dim: int, dropout: float) -> None:
        super().__init__()

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.do(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        x = x.permute(0, 2, 1)
        return x


@LayerFactory.register("ammvf_decoder_layer")
class DecoderLayer(nn.Module):
    def __init__(self, hid_dim: int, n_heads: int, pf_dim: int, self_attention: str, cross_attention: str, feed_forward: str, dropout: float) -> None:
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = LayerFactory.create(
            self_attention, hid_dim, n_heads, **{"drop": dropout})
        self.ea = LayerFactory.create(
            cross_attention, hid_dim, n_heads, **{"drop": dropout})
        self.pf = LayerFactory.create(feed_forward, hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        trg1 = self.ln(trg + self.do(self.sa(trg, mask=trg_mask)))
        trg1 = self.ln(trg1 + self.do(self.ea(trg1, src, mask=src_mask)))
        trg1 = self.ln(trg1 + self.do(self.pf(trg1)))
        src1 = self.ln(src + self.do(self.sa(src, mask=src_mask)))
        src1 = self.ln(src1 + self.do(self.ea(src1, trg, mask=trg_mask)))
        src1 = self.ln(src1 + self.do(self.pf(src1)))

        return trg1, src1


@LayerFactory.register("ammvf_decoder_block")
class Decoder(nn.Module):
    def __init__(self, hid_dim: int, n_layers: int, n_heads: int, pf_dim: int, decoder_layer: str, self_attention: str, cross_attention: str, feed_forward: str, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [LayerFactory.create(decoder_layer, hid_dim, n_heads, pf_dim, self_attention, cross_attention, feed_forward, dropout)
             for _ in range(n_layers)])

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        for layer in self.layers:
            trg, src = layer(trg, src, trg_mask, src_mask)

        trg = torch.mean(trg, dim=1)
        src = torch.mean(src, dim=1)
        return trg, src


@LayerFactory.register("ammvf_interaction_block")
class InteractionModel(nn.Module):
    def __init__(self, hid_dim, n_heads, attention_layer):
        super(InteractionModel, self).__init__()
        self.compound_attention = LayerFactory.create(
            attention_layer, hid_dim, n_heads)
        self.protein_attention = LayerFactory.create(
            attention_layer, hid_dim, n_heads)
        self.compound_fc = nn.Linear(hid_dim, hid_dim)
        self.protein_fc = nn.Linear(hid_dim, hid_dim)
        self.activation = nn.ReLU()

        self.hid_dim = hid_dim

    def forward(self, compound_features, protein_features):
        compound_embedded = self.activation(compound_features)
        protein_embedded = self.activation(protein_features)

        compound_embedded = compound_embedded.permute(1, 0, 2)
        protein_embedded = protein_embedded.permute(1, 0, 2)

        compound_attention_output, _ = self.compound_attention(
            compound_embedded, return_attn=True)
        protein_attention_output, _ = self.protein_attention(
            protein_embedded, return_attn=True)

        compound_attention_output = compound_attention_output.permute(1, 0, 2)
        protein_attention_output = protein_attention_output.permute(1, 0, 2)

        compound_output = self.activation(
            self.compound_fc(compound_attention_output))
        protein_output = self.activation(
            self.protein_fc(protein_attention_output))

        com_att = torch.unsqueeze(torch.mean(compound_output, 1), 1)
        pro_att = torch.unsqueeze(torch.mean(protein_output, 1), 1)
        return com_att, pro_att


@LayerFactory.register("ammvf_tensor_network_block")
class TensorNetworkModule(torch.nn.Module):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """

    def __init__(self, k_feature, hid_dim, k_dim):
        super(TensorNetworkModule, self).__init__()
        self.k_feature = k_feature
        self.hid_dim = hid_dim
        self.k_dim = k_dim

        self.setup_weights()
        self.init_parameters()

        self.fc1 = nn.Linear(hid_dim, k_dim)
        self.fc2 = nn.Linear(k_dim, hid_dim)

    def setup_weights(self):
        """
        Defining weights.  k_feature = args.filters_3   args.tensor_neurons = k_dim
        """
        self.weight_matrix = torch.nn.Parameter(
            torch.Tensor(
                self.k_feature, self.k_feature, self.k_dim
            )
        )
        self.weight_matrix_block = torch.nn.Parameter(
            torch.Tensor(self.k_dim, 2 * self.k_feature)
        )
        self.bias = torch.nn.Parameter(torch.Tensor(self.k_dim, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.   com_att
        :param embedding_2: Result of the 2nd embedding after attention.   pro_att
        :return scores: A similarity score vector.
        """
        embedding_1 = torch.squeeze(embedding_1, dim=1)
        embedding_1 = self.fc1(embedding_1)
        embedding_2 = torch.squeeze(embedding_2, dim=1)
        embedding_2 = self.fc1(embedding_2)

        batch_size = len(embedding_1)
        scoring = torch.matmul(
            embedding_1, self.weight_matrix.view(self.k_feature, -1)
        )
        scoring = scoring.view(
            batch_size, self.k_feature, -1).permute([0, 2, 1])
        scoring = torch.matmul(
            scoring, embedding_2.view(batch_size, self.k_feature, 1)
        ).view(batch_size, -1)
        combined_representation = torch.cat((embedding_1, embedding_2), 1)
        block_scoring = torch.t(
            torch.mm(self.weight_matrix_block,
                     torch.t(combined_representation))
        )
        scores = F.relu(scoring + block_scoring + self.bias.view(-1))
        scores = torch.unsqueeze(scores, 1)
        scores = self.fc2(scores)
        return scores


@ModelFactory.register("ammvf")
class AMMVF(nn.Module):
    def __init__(self,
                 n_fingerprint: int,
                 atom_dim: int,
                 hidden_dim: int,
                 protein_dim: int,
                 ligand_graph_conv_layer: Sequence[str],
                 ligand_graph_conv_layer_args: Sequence[dict],
                 ligand_graph_conv_layer_dims: Sequence[int],
                 ligand_conv_dropout_rate: Sequence[float],
                 ligand_conv_normalization: Sequence[str],
                 encoder_block: str,
                 encoder_block_args: dict,
                 decoder_block: str,
                 decoder_block_args: dict,
                 head_dims: Sequence[int],
                 head_dropout: Sequence[float],
                 head_activation: Sequence[str],
                 inter_attention_block: str,
                 inter_attention_block_args: dict,
                 tensor_network_block: str,
                 tensor_network_block_args: dict):
        """
            AMMVF (Attention Mechanism and Multi-View Fusion-based Drug-Target Interaction) is a model for predicting drug-target interactions. It leverages an attention mechanism to focus on relevant features and a multi-view fusion approach to integrate heterogeneous data sources.

            This implementation corresponds to the model described in the paper "AMMVF-DTI: A Novel Model Predicting Drug–Target Interactions Based on Attention Mechanism and Multi-View Fusion." It uses graph convolutional networks to process drug representations and sequence encoders for protein features, with attention layers to weigh their importance for interaction prediction.

            Parameters:
                n_fingerprint (int): Number of fingerprints.
                atom_dim (int): Dimension of atom embeddings.
                hidden_dim (int): Dimension of hidden layers.
                protein_dim (int): Dimension of protein embeddings.
                ligand_graph_conv_layer (Sequence[str]): Sequence of names for ligand graph convolutional layers.
                ligand_graph_conv_layer_args (Sequence[dict]): Arguments for ligand graph convolutional layers.
                ligand_graph_conv_layer_dims (Sequence[int]): Dimension specifications for ligand graph convolutional layers.
                ligand_conv_dropout_rate (Sequence[float]): Dropout rates for ligand convolutional layers.
                ligand_conv_normalization (Sequence[str]): Normalization types for ligand convolutional layers.
                encoder_block (str): Name of the encoder block.
                encoder_block_args (dict): Arguments for the encoder block.
                decoder_block (str): Name of the decoder block.
                decoder_block_args (dict): Arguments for the decoder block.
                head_dims (Sequence[int]): Sequence of dimensions for head layers.
                head_dropout (Sequence[float]): Dropout rates for head layers.
                head_activation (Sequence[str]): Activation functions for head layers.
                inter_attention_layer (str): Name of the inter-attention layer.
                inter_attention_layer_args (dict): Arguments for the inter-attention layer.
                inter_attention_block (str): Name of the inter-attention block.
                inter_attention_block_args (dict): Arguments for the inter-attention block.
                tensor_network_block (str): Name of the tensor network block.
                tensor_network_block_args (dict): Arguments for the tensor network block.
        """

        super().__init__()

        self.embed_fingerprint = nn.Embedding(n_fingerprint, atom_dim)

        # Ensure the dimensions sequence is correct
        ligand_graph_conv_layer_dims = [
            atom_dim] + ligand_graph_conv_layer_dims
        assert len(ligand_graph_conv_layer_dims) == len(ligand_graph_conv_layer) == len(
            ligand_conv_dropout_rate) == len(ligand_conv_normalization) == len(ligand_graph_conv_layer_args)

        # Create ligand graph convolution layers
        self.ligand_graph_conv_layer = nn.ModuleList([
            LayerFactory.create(ligand_graph_conv_layer[i],
                                ligand_graph_conv_layer_dims[i],
                                ligand_graph_conv_layer_dims[i],
                                normalization=ligand_conv_normalization[i],
                                dropout=ligand_conv_dropout_rate[i],
                                **ligand_graph_conv_layer_args[i]) for i in range(len(ligand_graph_conv_layer))
        ])

        self.encoder = LayerFactory.create(encoder_block, **encoder_block_args)
        self.decoder = LayerFactory.create(decoder_block, **decoder_block_args)

        # Inter-attention and tensor network blocks
        inter_attention_block_args['hid_dim'] = hidden_dim
        self.inter_att = LayerFactory.create(
            inter_attention_block, **inter_attention_block_args)

        tensor_network_block_args['hid_dim'] = hidden_dim
        self.tensor_network = LayerFactory.create(
            tensor_network_block, **tensor_network_block_args)

        # Initialize weights
        self.weight = nn.Parameter(torch.FloatTensor(atom_dim, atom_dim))
        self.init_weights()

        # Fully connected layers
        self.protein_dim = protein_dim
        self.hid_dim = hidden_dim
        self.atom_dim = atom_dim
        self.fc1 = nn.Linear(self.protein_dim, self.hid_dim)
        self.fc2 = nn.Linear(self.atom_dim, self.hid_dim)

        # Attention weights
        self.W_attention = nn.Linear(self.hid_dim, self.hid_dim)

        # Head layers for final prediction
        self.head_dims = list(head_dims)
        self.head = nn.Sequential(*[
            nn.Sequential(nn.Linear(self.head_dims[i], self.head_dims[i + 1]),
                          ActivationFactory.create(head_activation[i]),
                          nn.Dropout(head_dropout[i])) for i in range(len(self.head_dims) - 1)
        ])

    def init_weights(self) -> None:
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, protein1: Tensor, protein2: Tensor, compound2: Tensor, g: Any) -> Tensor:
        protein1 = torch.unsqueeze(protein1, dim=0)
        protein1 = self.fc1(protein1)
        compound1 = get_node_attr(g)
        compound1 = torch.unsqueeze(compound1, dim=0)
        compound1 = compound1.to(torch.float64)
        compound1 = self.fc2(compound1)
        protein1_c, compound1_p = self.decoder(protein1, compound1)
        compound2 = compound2.to(torch.long)
        compound2 = self.embed_fingerprint(compound2)
        g_new = change_node_attr(g, compound2)

        for layer in self.ligand_graph_conv_layer:
            g_new = layer(g_new)
            g_new = change_node_attr(g_new, get_node_attr(g_new).mean(dim=1))

        compound2 = get_node_attr(g_new)
        compound2 = self.fc2(compound2.unsqueeze(0))
        protein2 = self.encoder(protein2.unsqueeze(0))
        com_att, pro_att = self.inter_att(compound2, protein2)

        scores = self.tensor_network(com_att, pro_att).squeeze(1)
        out_fc = torch.cat((scores, compound1_p, protein1_c), 1)

        out = self.head(out_fc)

        return out
