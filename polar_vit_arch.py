import torch
import torch.nn as nn
import math
from torch.nn import functional as F

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )

class Config:
    def __init__(self):
        self.hidden_size = 1024
        self.num_attention_heads = 4
        self.num_layers = 2

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.layernorm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return self.layernorm(output + residual)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads, max_seq_len=6000, max_batch_size=1):
        super().__init__()

        self.n_kv_heads = n_heads
        self.n_heads_q = n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads

        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(
        self,
        x_q,
        x_k,
        x_v,
        freqs_complex,
        mask,
        rope
    ):
        batch_size, seq_len, _ = x_q.shape  # (B, 1, Dim)

        xq = self.wq(x_q)
        xk = self.wk(x_k)
        xv = self.wv(x_v)

        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Repeat k, v for matching q dimensions
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores += mask

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        output = torch.matmul(scores, xv)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        output = self.wo(output)

        return output, scores # (B, 1, Dim) -> (B, 1, Dim)
    
class Decoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads) for _ in range(n_layers)])
        self.d_model = d_model
        self.n_heads = n_heads

    def forward(self, dec_inputs, enc_outputs, mask=None):
        dec_outputs = dec_inputs

        dec_enc_attns = []
        for layer in self.layers:
            dec_outputs, dec_enc_attn = layer(dec_outputs, enc_outputs)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_enc_attns
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.dec_enc_attn = MultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_model * 4)

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)

    def forward(self, dec_inputs, enc_outputs, freqs_complex=None, mask=None):
        # Masked Self-Attention with Residual Connection
        dec_outputs = dec_inputs
        dec_outputs, _ = self.self_attn(dec_inputs, dec_inputs, dec_inputs, freqs_complex, mask=mask, rope=True)
        dec_outputs = self.layernorm1(dec_outputs + dec_inputs)  # Residual Connection

        # Encoder-Decoder Attention with Residual Connection
        dec_outputs1, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, freqs_complex, mask=mask, rope=True)
        dec_outputs = self.layernorm2(dec_outputs + dec_outputs1)  # Residual Connection

        # Position-wise Feed-Forward with Residual Connection
        dec_outputs2 = self.pos_ffn(dec_outputs)
        dec_outputs = self.layernorm3(dec_outputs + dec_outputs2)  # Residual Connection

        # [batch_size, tgt_len, d_model], [batch_size, h_heads, tgt_len, src_len]
        return dec_outputs, dec_enc_attn

class PolarViT(nn.Module):
    def __init__(self, vit1, vit2, decoder_1, decoder_2, classification_head, use_alpha=True, use_linear=False, freeze_initial_layers=True):
        super(PolarViT, self).__init__()
        self.vit1 = vit1
        self.vit2 = vit2
        self.decoder_1 = decoder_1
        self.decoder_2 = decoder_2
        self.classification_head = classification_head
        self.class_token1 = self.vit1.class_token
        self.class_token2 = self.vit2.class_token
        self.use_alpha = use_alpha
        self.use_linear = use_linear
        self.freeze_initial_layers = freeze_initial_layers
        
        # Assert that at least one of use_alpha or use_linear is True
        assert not (use_alpha and use_linear) and (use_alpha or use_linear)
        if self.use_alpha:
            print("Using alpha weighted approach for fusion of features")
            self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        elif self.use_linear:
            print("Using linear layer approach for fusion of features")
            self.linear_features = nn.Linear(2048, 1024)

        if self.freeze_initial_layers:
            self.freeze_initial_layers_func(self.vit1)
            self.freeze_initial_layers_func(self.vit2)

    def freeze_initial_layers_func(self, model):
        for param in model.conv_proj.parameters():
            param.requires_grad = False
        for param in model.encoder.parameters():
            param.requires_grad = False

    def forward(self, x1, x2):
        x1 = (self.vit1._process_input(x1))  # Features from first ViT
        n = x1.shape[0]
        batch_class_token1 = self.class_token1.expand(n, -1, -1)
        x1 = torch.cat((batch_class_token1, x1), dim=1)
        x2 = (self.vit2._process_input(x2))  # Features from second ViT
        n = x2.shape[0]
        batch_class_token2 = self.class_token2.expand(n, -1, -1)
        x2 = torch.cat((batch_class_token2, x2), dim=1)
        features1 = self.vit1.encoder(x1)
        features2 = self.vit2.encoder(x2)
        # Decoder 1
        features1, _ = self.decoder_1(features1, features2)
        # Decoder 2
        features2, _ = self.decoder_2(features2, features1)
        if self.use_alpha:
            alpha = torch.sigmoid(self.alpha)
            # Combine features from both ViTs using the alpha parameter
            combined_features = alpha * features1 + (1 - alpha) * features2
            class_token = combined_features[:, 0] 
        elif self.use_linear:
            # Combine features from both ViTs using a linear layer
            combined_features = torch.cat((features1, features2), dim=2)
            processed_combined_features = self.linear_features(combined_features)
            class_token = processed_combined_features[:, 0]
        return self.classification_head(class_token)