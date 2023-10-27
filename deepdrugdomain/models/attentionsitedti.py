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
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from deepdrugdomain.layers import LayerFactory, ActivationFactory
from typing import Optional, Sequence

from deepdrugdomain.utils.weight_init import trunc_normal_
from .factory import ModelFactory


@ModelFactory.register("attentionsitedti")
class AttentionSiteDTI(nn.Module):
    """
        An Interpretable Graph-Based Model for Drug-Target Interaction (DTI) Prediction.

        The AttentionSiteDTI model is a sophisticated deep learning architecture that represents 
        both drugs and targets as graphs. It leverages graph convolutional layers for feature 
        extraction, an attention mechanism to emphasize crucial interaction regions, and optionally, 
        LSTM layers to account for sequential patterns in the combined drug-target representation.

        Attributes:
        ------------
        protein_graph_conv : nn.ModuleList
            List of graph convolutional layers dedicated to processing protein representations.

        ligand_graph_conv : nn.ModuleList
            List of graph convolutional layers for processing ligand representations.

        pool_protein : nn.Module
            Pooling layer for protein graph representations.

        pool_ligand : nn.Module
            Pooling layer for ligand graph representations.

        lstm : nn.Module
            LSTM layers for capturing sequential patterns. Used if `use_lstm_layer` is True.

        attention : nn.Module
            Attention mechanism layer emphasizing regions critical for drug-target interaction.

        fc : nn.ModuleList
            List of fully connected layers leading to the prediction layer.

        fc_out : nn.Linear
            The final prediction layer.

        activation : nn.Module
            Activation function applied after the fully connected layers.

        Parameters:
        ------------
        protein_graph_conv_layer : str
            Specifies the type of graph convolutional layer to be used for processing protein graph representations.

        ligand_graph_conv_layer : str
            Specifies the type of graph convolutional layer to be used for processing ligand (drug) graph representations.

        protein_input_size : int
            Initial dimensionality of the input features for proteins.

        ligand_input_size : int
            Initial dimensionality of the input features for ligands (drugs).

        protein_graph_conv_dims : Sequence[int]
            Dimensions for each subsequent graph convolutional layer dedicated to protein representations.

        ligand_graph_conv_dims : Sequence[int]
            Dimensions for each subsequent graph convolutional layer dedicated to ligand representations.

        sequence_length : int
            Expected length of the combined drug-target sequence representation.

        embedding_dim : int
            Desired dimensionality of the embeddings after combining drug and target representations.

        ligand_graph_pooling : str
            Defines the type of graph pooling mechanism to be used on ligand graphs.

        protein_graph_pooling : str
            Defines the type of graph pooling mechanism to be used on protein graphs.

        use_lstm_layer : bool
            Flag to decide if LSTM layers should be used in the model for capturing sequential patterns.

        use_bilstm : bool
            If set to True, a bidirectional LSTM will be used. Relevant only if `use_lstm_layer` is True.

        lstm_input : Optional[int]
            Size of input features for the LSTM layer.

        lstm_output : Optional[int]
            Output size from the LSTM layer.

        lstm_num_layers : Optional[int]
            Number of LSTM layers to be used.

        lstm_dropout_rate : Optional[float]
            Dropout rate to be applied to LSTM layers.

        head_dims : Sequence[int]
            Defines the dimensions for each subsequent fully connected layer leading to the final prediction.

        attention_layer : str
            Specifies the type of attention layer to be used in the model.

        attention_head : int
            Number of attention heads in the attention mechanism.

        attention_dropout : float
            Dropout rate applied in the attention mechanism.

        qk_scale : Optional[float]
            Scaling factor for the query-key dot product in the attention mechanism.

        proj_drop : float
            Dropout rate applied after the projection in the attention mechanism.

        attention_layer_bias : bool
            If True, biases will be included in the attention mechanism computations.

        protein_conv_dropout_rate : Sequence[float]
            Dropout rates for each protein graph convolutional layer.

        protein_conv_normalization : Sequence[str]
            Normalization types (e.g., 'batch', 'layer') for each protein graph convolutional layer.

        ligand_conv_dropout_rate : Sequence[float]
            Dropout rates for each ligand graph convolutional layer.

        ligand_conv_normalization : Sequence[str]
            Normalization types (e.g., 'batch', 'layer') for each ligand graph convolutional layer.

        head_dropout_rate : float
            Dropout rate applied before the final prediction layer.

        head_activation_fn : Optional[str]
            Activation function applied after the fully connected layers. If None, no activation is applied.

        protein_graph_conv_kwargs : Sequence[dict]
            Additional keyword arguments for each protein graph convolutional layer.

        ligand_graph_conv_kwargs : Sequence[dict]
            Additional keyword arguments for each ligand graph convolutional layer.

        ligand_graph_pooling_kwargs : dict
            Additional keyword arguments for the ligand graph pooling layer.

        protein_graph_pooling_kwargs : dict
            Additional keyword arguments for the protein graph pooling layer.

        **kwargs : 
            Other additional keyword arguments not explicitly listed above.

    """

    def __init__(self,
                 protein_graph_conv_layer: str,
                 ligand_graph_conv_layer: str,
                 protein_input_size: int,
                 ligand_input_size: int,
                 protein_graph_conv_dims: Sequence[int],
                 ligand_graph_conv_dims: Sequence[int],
                 sequence_length: int,
                 embedding_dim: int,
                 ligand_graph_pooling: str,
                 protein_graph_pooling: str,
                 use_lstm_layer: bool,
                 use_bilstm: bool,
                 lstm_input: Optional[int],
                 lstm_output: Optional[int],
                 lstm_num_layers: Optional[int],
                 lstm_dropout_rate: Optional[float],
                 head_dims: Sequence[int],
                 attention_layer: str,
                 attention_head: int,
                 attention_dropout: float,
                 qk_scale: Optional[float],
                 proj_drop: float,
                 attention_layer_bias: bool,
                 protein_conv_dropout_rate: Sequence[float],
                 protein_conv_normalization: Sequence[str],
                 ligand_conv_dropout_rate: Sequence[float],
                 ligand_conv_normalization: Sequence[str],
                 head_dropout_rate: float,
                 head_activation_fn: Optional[str],
                 protein_graph_conv_kwargs: Sequence[dict],
                 ligand_graph_conv_kwargs: Sequence[dict],
                 ligand_graph_pooling_kwargs: dict,
                 protein_graph_pooling_kwargs: dict,
                 **kwargs
                 ) -> None:
        """Initialize the AttentionSiteDTI model."""
        super(AttentionSiteDTI, self).__init__()
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length

        # Initialize protein graph convolution layers
        p_dims = [protein_input_size] + list(protein_graph_conv_dims)
        assert len(p_dims) - 1 == len(protein_conv_dropout_rate) == len(protein_graph_conv_kwargs) == len(
            protein_conv_normalization), "The number of protein graph convolution layers parameters must be the same"
        self.protein_graph_conv = nn.ModuleList([
            LayerFactory.create(protein_graph_conv_layer,
                                p_dims[i],
                                p_dims[i + 1],
                                normalization=protein_conv_normalization[i],
                                dropout=protein_conv_dropout_rate[i],
                                **protein_graph_conv_kwargs[i]) for i in range(len(p_dims) - 1)])

        # Initialize drug graph convolution layers
        l_dims = [ligand_input_size] + list(ligand_graph_conv_dims)
        assert len(l_dims) - 1 == len(ligand_conv_dropout_rate) == len(ligand_graph_conv_kwargs) == len(
            ligand_conv_normalization), "The number of ligand graph convolution layers parameters must be the same"
        self.ligand_graph_conv = nn.ModuleList([
            LayerFactory.create(ligand_graph_conv_layer,
                                l_dims[i],
                                l_dims[i + 1],
                                normalization=ligand_conv_normalization[i],
                                dropout=ligand_conv_dropout_rate[i],
                                **ligand_graph_conv_kwargs[i]) for i in range(len(l_dims) - 1)])

        # Graph pooling layers
        self.pool_ligand = LayerFactory.create(
            ligand_graph_pooling, **ligand_graph_pooling_kwargs)
        self.pool_protein = LayerFactory.create(
            protein_graph_pooling, **protein_graph_pooling_kwargs)

        self.head_dropout = nn.Dropout(head_dropout_rate)

        # Graph pooling layers
        self.use_lstm = use_lstm_layer
        if use_lstm_layer:
            assert None not in [lstm_input, lstm_output, lstm_dropout_rate,
                                lstm_num_layers], "You need to set the LSTM parameters in the model"
            self.lstm = nn.LSTM(lstm_input, self.embedding_dim, num_layers=lstm_num_layers, bidirectional=use_bilstm,
                                dropout=lstm_dropout_rate)
            self.h_0 = Variable(torch.zeros(
                lstm_num_layers * 2, 1, self.embedding_dim).cuda())
            self.c_0 = Variable(torch.zeros(
                lstm_num_layers * 2, 1, self.embedding_dim).cuda())

        else:
            self.lstm = nn.Identity()

        assert self.embedding_dim % attention_head == 0, "The embedding dimension must be advisable by number of \
                                                          attention heads"

        # Attention layer
        self.attention = LayerFactory.create(attention_layer, self.embedding_dim, num_heads=attention_head,
                                             qkv_bias=attention_layer_bias, qk_scale=qk_scale,
                                             attn_drop=attention_dropout, proj_drop=proj_drop, **kwargs)

        # Prediction layer
        self.fc = nn.ModuleList()
        neuron_list = [self.embedding_dim * sequence_length] + list(head_dims)
        for item in range(len(neuron_list) - 2):
            self.fc.append(nn.Linear(neuron_list[item], neuron_list[item + 1]))

        self.fc_out = nn.Linear(neuron_list[-2], neuron_list[-1])

        self.activation = ActivationFactory.create(
            head_activation_fn) if head_activation_fn else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
            Initialize weights for the layers.

            This method applies a truncated normal initialization for linear layers and 
            sets up biases/weights for LayerNorm layers.

            Parameters:
            - m (nn.Module): A PyTorch module whose weights need to be initialized.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, g):
        """
            Extract features from drug and target graphs.

            This method processes the drug and target graphs through their respective 
            graph convolutional layers and pools the results.

            Parameters:
            - g (tuple): A tuple containing the drug and target graphs.

            Returns:
            - Tuple[Tensor, Tensor]: Protein and ligand representations after convolution and pooling.
        """
        feature_protein = g[0].ndata['h']
        feature_smile = g[1].ndata['h']
        for module in self.protein_graph_conv:
            feature_protein = module(g[0], feature_protein)

        for module in self.ligand_graph_conv:
            feature_smile = module(g[1], feature_smile)

        protein_rep = self.pool_protein(
            g[0], feature_protein).view(-1, self.embedding_dim)
        ligand_rep = self.pool_ligand(
            g[1], feature_smile).view(-1, self.embedding_dim)

        return protein_rep, ligand_rep

    def generate_attention_mask(self, sequence_size):
        """
            Generate an attention mask based on sequence size.

            This mask is designed to pay attention to valid parts of the sequence and ignore padding.

            Parameters:
            - sequence_size (int): Size of the valid sequence.

            Returns:
            - Tensor: The generated attention mask.
        """
        mask = torch.eye(self.sequence_length, dtype=torch.uint8).view(1, self.sequence_length,
                                                                       self.sequence_length)
        mask[0, sequence_size:self.sequence_length, :] = 0
        mask[0, :, sequence_size:self.sequence_length] = 0
        mask[0, :, sequence_size - 1] = 1
        mask[0, sequence_size - 1, :] = 1
        mask[0, sequence_size - 1, sequence_size - 1] = 0

        return mask

    def forward(self, g):
        """
            Forward pass of the model.

            Process the drug and target graphs through the network, making use of the attention mechanism, 
            and optionally, LSTM layers, to predict their interaction.

            Parameters:
            - g (tuple): A tuple containing the drug and target graphs.

            Returns:
            - Tuple[Tensor, Tensor]: Model output and attention weights.
        """

        protein_rep, ligand_rep = self.forward_features(g)
        sequence = torch.cat((ligand_rep, protein_rep), dim=0).view(
            1, -1, self.embedding_dim)
        mask = self.generate_attention_mask(
            sequence.size()[1]).to(sequence.device)
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
        for layer in self.fc:
            out = F.relu(layer(out))
            out = self.head_dropout(out)
        out = torch.sigmoid(self.fc_out(out))
        return out
