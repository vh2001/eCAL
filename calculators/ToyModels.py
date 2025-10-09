import torch.nn as nn
import torch.nn.functional as F

import torch
import numpy as np
class SimpleMLP(nn.Module):
    def __init__(self, input_size=10, hidden_size=10, output_size=2, num_layers=3):
        super(SimpleMLP, self).__init__()
        
        # Create a ModuleList to store variable number of layers
        self.layers = nn.ModuleList()
        
        # First layer (input to hidden)
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
        
        # Output layer
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Pass through all layers sequentially
        for layer in self.layers:
            x = layer(x)
        return self.output(x)
class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=10, output_size=2, num_layers=3):
        super(SimpleCNN, self).__init__()
        
        self.layers = nn.ModuleList()
        current_channels = input_channels
        
        # Create convolutional layers
        for i in range(num_layers):
            # More controlled channel growth
            out_channels = hidden_channels * (2 if i > 0 else 1)
            
            # Create conv block with standard pooling
            conv_block = nn.Sequential(
                nn.Conv1d(current_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            )
            
            self.layers.append(conv_block)
            current_channels = out_channels
        
        # Global average pooling
        self.global_pool = nn.AvgPool1d(kernel_size=2)
        
        # Output layer
        self.output = nn.Linear(current_channels, output_size)
    
    def forward(self, x):
        # Pass through all convolutional blocks
        for layer in self.layers:
            x = layer(x)
            
        
        # Ensure we have at least one feature
        if x.size(-1) > 1:
            x = self.global_pool(x)
        
        # Global average pooling
        x = torch.mean(x, dim=-1)
        
        return self.output(x)
    




class SimpleMLP_practical(nn.Module):
    def __init__(self, input_size=10, hidden_size=10, output_size=2, num_layers=3):
        super(SimpleMLP_practical, self).__init__()
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.ReLU())
        # Hidden layers
        for _ in range(num_layers - 2): # Adjusted loop for clarity
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
        # Output layer - Add it to the list!
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        # Simpler forward pass that processes all layers sequentially
        for layer in self.layers:
            x = layer(x)
        return x


# --- CNN Model (Modified to vary dense layers) ---
class SimpleCNN_practical(nn.Module):
    """
    A simple 1D Convolutional Neural Network (CNN) for sequence or feature vector regression.
    """
    def __init__(self, output_size=2, hidden_channels=16, num_layers=3):
        super(SimpleCNN_practical, self).__init__()
        self.layers = nn.ModuleList()
        # Input data is expected to be reshaped to have 1 channel.
        current_channels = 1  
        
        # Create convolutional blocks with increasing channel depth
        for i in range(num_layers):
            out_channels = hidden_channels * (2**i) # e.g., 16 -> 32 -> 64
            conv_block = nn.Sequential(
                nn.Conv1d(current_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )
            self.layers.append(conv_block)
            current_channels = out_channels
            
        # Global pooling layer adapts to any input size
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_layer = nn.Linear(current_channels, output_size)
    
    def forward(self, x):

        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension if missing
        # Expected input x shape: (batch_size, 1, num_features)
        for layer in self.layers:
            x = layer(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1) # Flatten the output for the linear layer
        return self.output_layer(x)
# --- Simplified KAN-like Model (Modified to vary sub-layers) ---
# This model applies an MLP with `num_hidden_layers_per_feature_fn` hidden layers
# to each input feature and sums the results.
class KANLikeRegressor(nn.Module):
    def __init__(self, input_dim, num_hidden_layers_per_feature_fn, nodes_per_layer_in_sub_fn):
        super(KANLikeRegressor, self).__init__()
        self.input_dim = input_dim
        self.feature_functions = nn.ModuleList()

        for _ in range(input_dim):
            layers_list = []
            current_in_features = 1 # Each feature function takes 1 input initially

            for i in range(num_hidden_layers_per_feature_fn):
                layers_list.append(nn.Linear(current_in_features, nodes_per_layer_in_sub_fn))
                layers_list.append(nn.ReLU())
                current_in_features = nodes_per_layer_in_sub_fn # For subsequent hidden layers

            # Final output layer for the feature function
            layers_list.append(nn.Linear(current_in_features, 1))
            self.feature_functions.append(nn.Sequential(*layers_list))

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        outputs = []
        for i in range(self.input_dim):
            # Pass each feature through its own sub-network
            output = self.feature_functions[i](x[:, i].unsqueeze(1)) # (batch_size, 1)
            outputs.append(output)
        # Sum the outputs of all feature functions
        return torch.sum(torch.cat(outputs, dim=1), dim=1, keepdim=True)


#############################TRANSFORMER MODEL FROM SCRATCH#############################

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_emb, num_heads=8):
        super().__init__()
        
        # hyperparams
        self.D = num_emb  # embedding size
        self.H = num_heads # number of transformer heads
        
        # weights for self-attention
        self.w_k = nn.Linear(self.D, self.D * self.H)
        self.w_q = nn.Linear(self.D, self.D * self.H)
        self.w_v = nn.Linear(self.D, self.D * self.H)
        
        # weights for a combination of multiple heads
        self.w_c = nn.Linear(self.D * self.H, self.D)
            
    def forward(self, x, causal=True):
        # x: B(atch) x T(okens) x D(imensionality)
        B, T, D = x.size()
        
        # keys, queries, values
        k = self.w_k(x).view(B, T, self.H, D) # B x T x H x D ########## K = x*W_k + b_k 
        q = self.w_q(x).view(B, T, self.H, D) # B x T x H x D ########## Q = x*W_q + b_q 
        v = self.w_v(x).view(B, T, self.H, D) # B x T x H x D ########## V = x*W_v + b_v 
        
        # batches and heads are merged for more efficent matrix multiplication
        # B x T x H x D -> B*H x T x D
        k = k.transpose(1, 2).contiguous().view(B * self.H, T, D) # B*H x T x D 
        q = q.transpose(1, 2).contiguous().view(B * self.H, T, D) # B*H x T x D
        v = v.transpose(1, 2).contiguous().view(B * self.H, T, D) # B*H x T x D
        
        k = k / (D**0.25) # scaling with sqrt(D) 
        q = q / (D**0.25) # scaling with sqrt(D)
        
        # kq
        kq = torch.bmm(q, k.transpose(1, 2)) # B*H x T x T # (Q x K^T) / sqrt(D)
        
        # if causal apply mask to prevent information flow from future tokens we set tokens above the diagonal to -inf so after softmax they are 0
        if causal:
            mask = torch.triu_indices(T, T, offset=1)
            kq[..., mask[0], mask[1]] = float('-inf')
        
        # softmax
        skq = F.softmax(kq, dim=2) # B*H x T x T | A = softmax((Q x K^T)/sqrt(D))
        
        # self-attention
        sa = torch.bmm(skq, v) # B*H x T x D # (softmax(Q x K^T) x V)
        sa = sa.view(B, self.H, T, D) # B x H x T x D
        sa = sa.transpose(1, 2) # B x T x H x D
        sa = sa.contiguous().view(B, T, D * self.H) # B x T x D*H
        
        out = self.w_c(sa) # B x T x D
        
        return out      
    

class TransformerBlock(nn.Module):
    def __init__(self, num_emb, num_neurons, num_heads=4):
        super().__init__()
        
        # hyperparams
        self.D = num_emb
        self.H = num_heads
        self.neurons = num_neurons
        
        # components
        self.msha = MultiHeadSelfAttention(num_emb=self.D, num_heads=self.H)
        self.layer_norm1 = nn.LayerNorm(self.D)
        self.layer_norm2 = nn.LayerNorm(self.D)
        
        self.mlp = nn.Sequential(nn.Linear(self.D, self.neurons * self.D),
                                nn.GELU(),
                                nn.Linear(self.neurons * self.D, self.D))
    
    def forward(self, x, causal=True):
        # Multi-Head Self-Attention
        x_attn = self.msha(x, causal)
        # LayerNorm
        x = self.layer_norm1(x_attn + x)
        # MLP
        x_mlp = self.mlp(x)
        # LayerNorm
        x = self.layer_norm2(x_mlp + x)
        
        return x        
    

class LossFun(nn.Module):
    def __init__(self,):
        super().__init__()
        
        self.loss = nn.MSELoss()
    
    def forward(self, y_model, y_true, reduction='sum'):
        # y_model: B(atch) x T(okens) x V(alues)
        # y_true: B x T      
        B, T, V = y_model.size()
        
        y_model = y_model.view(B * T, V)
        y_true = y_true.view(B * T,)
        

        loss_matrix = self.loss(y_model, y_true) # B*T
        
        if reduction == 'sum':
            return torch.sum(loss_matrix)
        elif reduction == 'mean':
            loss_matrix = loss_matrix.view(B, T)
            return torch.mean(torch.sum(loss_matrix, 1))
        else:
            raise ValueError('Reduction could be either `sum` or `mean`.')
        
class Transformer(nn.Module):
    def __init__(self, num_tokens, num_token_vals, num_emb, num_neurons, num_heads=2, dropout_prob=0.1, num_blocks=10, device='cpu'):
        super().__init__()
    
        # hyperparams
        self.device = device
        self.num_tokens = num_tokens
        self.num_emb = num_emb
        self.num_blocks = num_blocks

        # FIX 2: Re-enable the positional embedding layer
        self.positional_embedding = nn.Embedding(num_tokens, num_emb)

        # This correctly projects each feature in the sequence to the embedding dimension
        self.embedding = nn.Linear(1, num_emb)

        # transformer blocks
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.transformer_blocks.append(TransformerBlock(num_emb=num_emb, num_neurons=num_neurons, num_heads=num_heads))

        # Output layer for regression (predicting 2 values: target_x, target_y)
        regression_output_dim = 1
        self.output_layer = nn.Sequential(nn.Linear(num_emb * num_tokens, regression_output_dim))

        # dropout layer
        self.dropout = nn.Dropout(dropout_prob)


    def transformer_forward(self, x, causal=True):
        # x starts as: B(atch) x T(okens) -> (32, 25)

        # FIX 1: Add a dimension to the input for the linear embedding layer
        # x becomes: B x T x 1 -> (32, 25, 1)
        x = x.unsqueeze(-1)

        # embedding of tokens
        # x becomes B x T x D(im) -> (32, 25, 20)
        x = self.embedding(x)

        # embedding of positions
        # Create positions tensor: (1, 25)
        pos = torch.arange(0, x.shape[1], dtype=torch.long).unsqueeze(0).to(self.device)
        # Get positional embeddings: (1, 25, 20)
        pos_emb = self.positional_embedding(pos)

        # Add positional embeddings to input embeddings
        x = self.dropout(x + pos_emb)

        # transformer blocks
        for i in range(self.num_blocks):
            x = self.transformer_blocks[i](x, causal=False) # Causal might not be needed for this task

        # Flatten the output of the transformer blocks
        # x shape from (B, T, D) to (B, T * D) -> (32, 25 * 20)
        x = x.view(x.size(0), -1)

        # output layer
        out = self.output_layer(x)

        return out

    def forward(self, x, causal=True):
        # This method just calls the main forward pass
        return self.transformer_forward(x, causal=causal)