"""
Fair-Fed-CI: Privacy-Preserving Educational Prediction via Federated Learning
-------------------------------------------------------------------------
Model Architecture Module: Defines the EnhancedNet (with Feature and Self Attention)
and VanillaMLP architectures used by the federated clients and global server.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAttention(nn.Module):
    """
    Learns a weight for each input feature.
    Output: weighted_features, attention_weights
    """
    def __init__(self, input_dim):
        super(FeatureAttention, self).__init__()
        # A simple linear layer to score each feature
        self.attention_score = nn.Linear(input_dim, input_dim)
        
    def forward(self, x):
        # Calculate attention scores
        scores = self.attention_score(x)
        # Normalize scores to [0, 1] using Sigmoid (independent importance) 
        # or Softmax (relative importance). Sigmoid is often better for tabular feature selection.
        weights = torch.sigmoid(scores)
        
        # Apply weights
        weighted_features = x * weights
        return weighted_features, weights

import math

class SelfAttention(nn.Module):
    """自注意力机制，用于捕获特征间的依赖关系"""
    def __init__(self, input_dim, num_heads=4, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        
        # We enforce input_dim to be divisible by num_heads or handle padding
        if input_dim % num_heads != 0:
            self.padded_dim = ((input_dim // num_heads) + 1) * num_heads
            self.pad_layer = nn.ZeroPad2d((0, self.padded_dim - input_dim, 0, 0))
        else:
            self.padded_dim = input_dim
            self.pad_layer = nn.Identity()
            
        self.head_dim = self.padded_dim // num_heads
        
        self.query = nn.Linear(self.padded_dim, self.padded_dim)
        self.key = nn.Linear(self.padded_dim, self.padded_dim)
        self.value = nn.Linear(self.padded_dim, self.padded_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Projection back to original input_dim if padded
        self.out_proj = nn.Linear(self.padded_dim, input_dim)
        
    def forward(self, x):
        # x is (batch_size, input_dim), need to convert to sequence
        x = x.unsqueeze(1) # (batch, 1, dim)
        batch_size, seq_len, _ = x.size()
        
        x_pad = self.pad_layer(x)
        
        Q = self.query(x_pad).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x_pad).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x_pad).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.padded_dim)
        
        output = self.out_proj(context)
        output = output.squeeze(1) # (batch, dim)
        return output

class ResidualBlock(nn.Module):
    """残差连接块，包含批归一化和dropout"""
    def __init__(self, input_dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim)
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.layers(x)
        out = out + residual
        return self.relu(out)

class EnhancedNet(nn.Module):
    """
    Enhanced Deep Network serving as Core Model for Fair-Fed-CI v2.
    Architecture order:
    1. FeatureAttention (Shared)
    2. Input Linear Layer (Shared)
    3. ResidualBlock (Shared)
    4. SelfAttention (Shared)
    5. MLP Head (Personalized)
    """
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.1):
        super(EnhancedNet, self).__init__()
        
        self.feature_attention = FeatureAttention(input_dim)
        
        # 1. Input Layer connecting input_dim to hidden_dims[0]
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Residual Block
        self.residual_block = ResidualBlock(hidden_dims[0], dropout)
        
        # 3. Self Attention Layer
        self.self_attention = SelfAttention(hidden_dims[0], num_heads=4, dropout=dropout)
        
        # 4. Output MLP
        layers = []
        in_dim = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
            
        self.head_mlp = nn.Sequential(*layers)
        self.head_out = nn.Linear(in_dim, 1)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # 1. Feature Attention
        weighted_x, attn_weights = self.feature_attention(x)
        
        # 2. Representation processing
        out = self.input_layer(weighted_x)
        out = self.residual_block(out)
        out = self.self_attention(out)
        
        # 3. Prediction Head
        features = self.head_mlp(out)
        prediction = torch.sigmoid(self.head_out(features))
        
        return prediction, attn_weights

    def get_shared_parameters(self):
        """Return parameters for Shared Feature Attention, Residual Block and Self Attention."""
        params = {}
        # Everything except head_mlp and head_out goes to server
        for name, param in self.named_parameters():
            if not name.startswith("head_"):
                params[name] = param
        return params

    def get_personalized_parameters(self):
        """Return parameters for the Local personalized Head."""
        params = {}
        for name, param in self.named_parameters():
            if name.startswith("head_"):
                params[name] = param
        return params

class VanillaMLP(nn.Module):
    """
    A standard 3-layer MLP without attention or residual connections.
    Includes parameter splitting for federated personalization.
    """
    def __init__(self, input_dim):
        super(VanillaMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.relu(self.bn2(self.fc2(out)))
        prediction = torch.sigmoid(self.fc3(out))
        # Return dummy attention weights for compatibility
        dummy_attn = torch.zeros(x.shape[0], x.shape[1]).to(x.device)
        return prediction, dummy_attn

    def get_shared_parameters(self):
        # Only fc1 is shared
        return {"fc1.weight": self.fc1.weight, "fc1.bias": self.fc1.bias}

    def get_personalized_parameters(self):
        # fc2 and fc3 are personalized
        return {
            "fc2.weight": self.fc2.weight, "fc2.bias": self.fc2.bias,
            "fc3.weight": self.fc3.weight, "fc3.bias": self.fc3.bias
        }

if __name__ == "__main__":
    input_dim = 15
    batch_size = 5
    model = EnhancedNet(input_dim)
    
    dummy_input = torch.rand(batch_size, input_dim)
    output, weights = model(dummy_input)
    
    print("Model Output Shape:", output.shape)
    print("Attention Weights Shape:", weights.shape)
    print("Output Range:", output.min().item(), "-", output.max().item())
    
    print("\nShared Params Keys:", model.get_shared_parameters().keys())
    print("Personalized Params Keys:", model.get_personalized_parameters().keys())
