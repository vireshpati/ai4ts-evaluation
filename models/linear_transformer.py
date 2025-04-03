# models/linear_transformer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models import BaseFairseqModel, register_model, register_model_architecture
from fairseq.modules import LayerNorm

##############################
# linear attention and Transformer encoder layer
##############################
class LinearAttention(nn.Module):
    """
    Minimal linear attention with ELU+1 feature map.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        # x: [B, T, C]
        B, T, C = x.shape
        H = self.num_heads
        d = self.head_dim

        # project
        q = self.q_proj(x).view(B, T, H, d)
        k = self.k_proj(x).view(B, T, H, d)
        v = self.v_proj(x).view(B, T, H, d)
        # elu+1
        q = F.elu(q, inplace=False) + 1
        k = F.elu(k, inplace=False) + 1
        # possibly mask (not shown here for minimal code)
        # sum_{time} k_i * v_i
        kv = torch.einsum("bthd,bthe->bhde", k, v)  # shape [B, H, d, d]
        # out = q * (k^T * v) / (q * sum(k^T))
        # but more simply:
        denominator = torch.einsum("bthd,bhd->bth", q, k.sum(dim=1))  # [B,H,T]
        numerator = torch.einsum("bthd,bhdd->bthd", q, kv)            # [B,H,T,d]
        out = numerator / denominator.unsqueeze(-1).clamp(min=1e-6)   # [B,H,T,d]

        out = out.reshape(B, T, C)
        out = self.out_proj(out)
        return out

class LinearTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, ffn_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = LinearAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attn
        residual = x
        x_attn = self.attn(x)  # shape [B,T,embed_dim]
        x = residual + self.dropout(x_attn)
        x = self.norm1(x)
        # FFN
        residual = x
        x_ffn = self.ffn(x)
        x = residual + self.dropout(x_ffn)
        x = self.norm2(x)
        return x
    
##############################
# Model
##############################
@register_model("simple_linear_transformer")
class SimpleLinearTransformer(BaseFairseqModel):
    """
    Minimal Transformer using linear attention. 
    For classification tasks, we assume input is [B, T, C=1 or embed].
    We'll produce a [B, T, C] output, and rely on a classification head for final logits.
    """

    @staticmethod
    def add_args(parser):
        # Model hyperparameters
        parser.add_argument('--encoder-layers', type=int, default=4, help='num encoder layers')
        parser.add_argument('--encoder-embed-dim', type=int, default=256, help='embed dim')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, default=512, help='FFN dim')
        parser.add_argument('--encoder-attention-heads', type=int, default=4, help='num heads')
        parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    @classmethod
    def build_model(cls, args, task):
        """
        Build the model. The classification head will be added by the task if needed.
        """
        return cls(args)

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.num_layers = args.encoder_layers
        self.num_heads = args.encoder_attention_heads
        self.ffn_dim = args.encoder_ffn_embed_dim
        self.dropout = args.dropout

        self.layers = nn.ModuleList([
            LinearTransformerEncoderLayer(self.embed_dim, self.ffn_dim, self.num_heads, self.dropout)
            for _ in range(self.num_layers)
        ])
        self.layer_norm = LayerNorm(self.embed_dim)

    def forward(self, src_tokens, src_lengths=None, classification_head_name=None, **kwargs):
        """
        src_tokens: shape [B, T] or [B, T, C]
        Returns: tuple of (logits, None) where logits has shape [B, num_classes]
        """
        # Get features from encoder
        x = src_tokens  # [B, T] or [B, T, C]
        
        # Project if needed
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B,T,1]
            if not hasattr(self, 'input_proj'):
                self.input_proj = nn.Linear(1, self.embed_dim).to(x.device)
            x = self.input_proj(x)
        elif x.dim() == 3 and x.size(2) != self.embed_dim:
            if not hasattr(self, 'input_proj'):
                self.input_proj = nn.Linear(x.size(2), self.embed_dim).to(x.device)
            x = self.input_proj(x)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)  # [B,T,embed_dim]

        # For classification, use mean pooling over time dimension
        x = x.mean(dim=1)  # [B, embed_dim]

        # Pass through classification head if requested
        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)  # [B, num_classes]
        
        # Return in the format expected by Fairseq's cross entropy criterion
        return x, None

    def extract_features(self, src_tokens, src_lengths=None, **kwargs):
        """Extract features before classification head"""
        # If shape is [B,T], we project to [B,T,embed_dim]
        if src_tokens.dim() == 2:
            src_tokens = src_tokens.unsqueeze(-1)  # [B,T,1]
            proj = nn.Linear(1, self.embed_dim).to(src_tokens.device)
            src_tokens = proj(src_tokens)
        elif src_tokens.dim() == 3:
            if src_tokens.size(2) != self.embed_dim:
                proj = nn.Linear(src_tokens.size(2), self.embed_dim).to(src_tokens.device)
                src_tokens = proj(src_tokens)

        x = src_tokens  # [B, T, embed_dim]
        # pass through layers
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)  # [B,T,embed_dim]
        return x, None

    def get_normalized_probs_scriptable(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0]
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def classification_heads_forward(self, features, head_name):
        # features shape: [B,T,embed_dim], we take e.g. x[:,0,:]
        cls_repr = features[:,0,:]  # Use first token as CLS token
        head = self.classification_heads[head_name]
        return head(cls_repr)

    def register_classification_head(self, name, num_classes):
        # minimal approach: store a linear layer in a dict
        # But we won't do a full "head" approach for brevity. 
        # We'll do a single dictionary attribute
        if not hasattr(self, 'classification_heads'):
            self.classification_heads = nn.ModuleDict()
        self.classification_heads[name] = nn.Linear(self.embed_dim, num_classes)

    def to_device(self, device):
        """Move model to specified device"""
        self.input_proj = self.input_proj.to(device)
        self.layers = self.layers.to(device)
        self.layer_norm = self.layer_norm.to(device)
        if hasattr(self, 'classification_heads'):
            for head in self.classification_heads.values():
                head = head.to(device)
        return self

@register_model_architecture("simple_linear_transformer", "simple_linear_transformer_arch")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)

