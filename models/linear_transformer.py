# models/linear_transformer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models import BaseFairseqModel, register_model, register_model_architecture
from fairseq.modules import LayerNorm

##############################
# Minimal linear attention
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
        kv = torch.einsum("bthd,bthd->bhdd", k, v)  # shape [B, H, d, d]
        # out = q * (k^T * v) / (q * sum(k^T))
        # but more simply:
        denominator = torch.einsum("bthd,bhd->bth", q, k.sum(dim=1))  # [B,H,T]
        numerator = torch.einsum("bthd,bhdd->bthd", q, kv)            # [B,H,T,d]
        out = numerator / denominator.unsqueeze(-1).clamp(min=1e-6)   # [B,H,T,d]

        out = out.reshape(B, T, C)
        out = self.out_proj(out)
        return out

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
        src_tokens: shape [B, T], raw waveforms or [B, T, feats].
        If raw waveforms => we might do an input projection to embed_dim.
        For minimal code, assume we do it outside or in first layer.
        """
        # If shape is [B,T], we project to [B,T,embed_dim]
        if src_tokens.dim() == 2:
            # raw waveforms => project to embed_dim
            src_tokens = src_tokens.unsqueeze(-1)  # [B,T,1]
            proj = nn.Linear(1, self.embed_dim).to(src_tokens.device)
            src_tokens = proj(src_tokens)
        elif src_tokens.dim() == 3:
            # we assume second dim is T, third is feature dim
            if src_tokens.size(2) != self.embed_dim:
                # lazy creation of a projection layer if needed
                proj = nn.Linear(src_tokens.size(2), self.embed_dim).to(src_tokens.device)
                src_tokens = proj(src_tokens)
        else:
            raise ValueError("Unsupported input shape for SimpleLinearTransformer")

        x = src_tokens  # [B, T, embed_dim]
        # pass through layers
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)  # [B,T,embed_dim]

        # The classification head uses the 'CLS' position or average pool
        # We'll do a quick approach: take x[:,0,:] as the representation
        # The task can pass classification_head_name to request logits
        return x, None

    def register_classification_head(self, name, num_classes):
        # minimal approach: store a linear layer in a dict
        # But we won't do a full "head" approach for brevity. 
        # We'll do a single dictionary attribute
        if not hasattr(self, 'classification_heads'):
            self.classification_heads = nn.ModuleDict()
        self.classification_heads[name] = nn.Linear(self.embed_dim, num_classes)

    def classification_heads_forward(self, features, head_name):
        # features shape: [B,T,embed_dim], we take e.g. x[:,0,:]
        cls_repr = features[:,0,:]
        head = self.classification_heads[head_name]
        return head(cls_repr)

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

@register_model_architecture("simple_linear_transformer", "simple_linear_transformer_arch")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
