import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiplicativeAttention(nn.Module):
      
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        
        attn = torch.matmul(q , k.transpose(-2, -1) / math.sqrt(q.size(-1)))
        
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1) == 1, -1e9)
        
        attn = self.dropout(F.softmax(attn, dim=-1))        
        res = torch.matmul(attn, v)

        return res, attn

class AdditiveSelfAttention(nn.Module):
    
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        self.w = nn.Linear(d_model, d_model)
        self.q = torch.nn.Parameter(torch.FloatTensor(d_model).uniform_(-0.1, 0.1))
    
    def forward(self, x, mask=None):
        attn = torch.tanh(self.dropout(self.w(x)))        
        attn = torch.matmul(attn, self.q)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 1, -1e9)
        
        attn = self.dropout(F.softmax(attn, dim=-1))

        
        res = torch.einsum('ijk, ij->ik', x, attn)
        return res, attn

    
class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, num_heads, d_qk, d_v, track_agreement=False, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_qk = d_qk
        self.d_v = d_v
        
        self.num_heads = num_heads
        
        self.dropout = nn.Dropout(dropout)
        
        self.w_q = nn.Linear(d_model, num_heads * d_qk, bias=False)
        self.w_k = nn.Linear(d_model, num_heads * d_qk, bias=False)
        self.w_v = nn.Linear(d_model, num_heads * d_v, bias=False)
        
        self.w_fc = nn.Linear(num_heads * d_v, d_model, bias=False)
        
        self.attention = MultiplicativeAttention(dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
        self.track_agreement = track_agreement
        self.v_agreement = 0

    def forward(self, q, k, v, mask=None):     
        batch_size = q.shape[0]
        seq_size = q.shape[1]
        
        q_proj = self.w_q(q).view(q.shape[0], q.shape[1], self.num_heads, self.d_qk)
        k_proj = self.w_k(k).view(k.shape[0], k.shape[1], self.num_heads, self.d_qk)
        v_proj = self.w_v(v).view(v.shape[0], v.shape[1], self.num_heads, self.d_v) 

        if self.track_agreement:
        	self.v_agreement += torch.einsum('bshd, bsnd->', F.normalize(v_proj, dim=3), F.normalize(v_proj, dim=3)) / self.num_heads**2

        if mask is None:
            q, attn = self.attention(q_proj.transpose(1, 2), k_proj.transpose(1, 2), v_proj.transpose(1, 2))
        else:
            q, attn = self.attention(q_proj.transpose(1, 2), k_proj.transpose(1, 2), v_proj.transpose(1, 2), mask.unsqueeze(1))
        
        q = q.transpose(1, 2).contiguous()
        q = q.view(batch_size, seq_size, -1)

        q = self.dropout(self.w_fc(q))

        q = self.layer_norm(q)
        
        return q, attn

    def clear_agreement(self):
    	self.v_agreement = 0

class NonlinearFF(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)

        x = self.layer_norm(x)

        return x
    
class TitleEmbedding(nn.Module):
    def __init__(self, num_embeddings, d_model, num_heads, d_qk, d_v, d_hid=None, embeddings=None, track_agreement=False, padding_idx=0, dropout=0.1):
        super().__init__()

        if embeddings is not None:
            self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=False, sparse=True, padding_idx=padding_idx)
        else:
            self.embeddings = nn.Embedding(num_embeddings, d_model, sparse=True, padding_idx=0)
            
        self.mh_attn = MultiHeadAttention(d_model, num_heads, d_qk, d_v, track_agreement=track_agreement, dropout=dropout)
        self.nff = NonlinearFF(d_model, d_hid if d_hid is not None else d_model * 4, dropout=dropout)
        self.add_attn = AdditiveSelfAttention(d_model, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        self.padding_idx = padding_idx
        
    def forward(self, title):    
        mask = (title == self.padding_idx).byte()
        
        q = k = v = self.embeddings(title)
        title, attn = self.mh_attn(q, k ,v, mask=mask)
        
        title = self.nff(title)
        title, add_attn = self.add_attn(title, mask=mask)
        
        title = self.layer_norm(title)
        
        return title
    
    def load_embeddings(embeddings):
        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=False, sparse=True)
    