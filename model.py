import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from model_ae import Encoder



class MultiHeadedAttention(nn.Module):
    def __init__(self, num_attn_heads, attn_hidden_size, dropout_prob, with_focus_attn):
        super(MultiHeadedAttention, self).__init__()
        self.num_attn_heads = num_attn_heads
        self.hidden_size = attn_hidden_size
        self.dropout_prob = dropout_prob
        self.with_focus_attn = with_focus_attn
        
        self.attn_head_size = int(self.hidden_size / self.num_attn_heads)
        self.all_head_size = self.num_attn_heads * self.attn_head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.softmax = nn.Softmax(dim=-1)
        
        if(with_focus_attn == True):
            self.tanh = nn.Tanh()
            self.sigmoid = nn.Sigmoid()
            
            self.linear_focus_query = nn.Linear(num_attn_heads * self.attn_head_size, 
                                                num_attn_heads * self.attn_head_size)
            self.linear_focus_global = nn.Linear(num_attn_heads * self.attn_head_size, 
                                                 num_attn_heads * self.attn_head_size)
            
            up = torch.randn(num_attn_heads, 1, self.attn_head_size)
            self.up = Variable(up, requires_grad=True).cuda()
            torch.nn.init.xavier_uniform_(self.up)
            
            uz = torch.randn(num_attn_heads, 1, self.attn_head_size)
            self.uz = Variable(uz, requires_grad=True).cuda()
            torch.nn.init.xavier_uniform_(self.uz)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attn_heads, self.attn_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        key_len = hidden_states.size(1)
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        if(self.with_focus_attn == True):
            glo = torch.mean(mixed_query_layer, dim=1, keepdim=True)
            
            c = self.tanh(self.linear_focus_query(mixed_query_layer) + self.linear_focus_global(glo))
            c = self.transpose_for_scores(c)
            
            p = c * self.up
            p = p.sum(3).squeeze()
            z = c * self.uz
            z = z.sum(3).squeeze()
            
            P = self.sigmoid(p) * key_len
            Z = self.sigmoid(z) * key_len
            
            j = torch.arange(start=0, end=key_len, dtype=P.dtype).unsqueeze(0).unsqueeze(0).unsqueeze(0).to('cuda')
            P = P.unsqueeze(-1)
            Z = Z.unsqueeze(-1)
            
            G = -(j - P)**2 * 2 / (Z**2)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attn_head_size)
        
        if(self.with_focus_attn == True):
            attention_scores = attention_scores + G
            
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.o_proj(context_layer)

        return attention_output


class CLDNN(nn.Module):
    def __init__(self, conv_dim='1d', checkpoint=None, hidden_size=128, num_layers=2,
                 bidirectional=True, with_focus_attn=True):
        super(CLDNN, self).__init__()
        self.conv_dim = conv_dim
        if(conv_dim == '1d'):
            self.encoder = Encoder(conv_dim)
            if checkpoint:
                self.encoder.load_state_dict(torch.load(checkpoint))
            self.attn = MultiHeadedAttention(num_attn_heads=4, attn_hidden_size=8, dropout_prob=0.1,
                                             with_focus_attn=with_focus_attn)
            self.lstm = nn.LSTM(8, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size*2 if bidirectional else hidden_size, 1),
                nn.Sigmoid()
            )
        elif(conv_dim == '2d'):
            self.encoder = Encoder(conv_dim)
            if checkpoint:
                self.encoder.load_state_dict(torch.load(checkpoint))
            self.attn = MultiHeadedAttention(num_attn_heads=4, attn_hidden_size=176, dropout_prob=0.1, 
                                             with_focus_attn=with_focus_attn)
            self.gap = nn.AdaptiveAvgPool2d((1, 11))
            self.lstm = nn.LSTM(11, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size*2 if bidirectional else hidden_size, 1),
                nn.Sigmoid()
            )
        else:
            raise ValueError("Convolution dimension not found: %s" % (conv_dim))
            
    def forward(self, x):
        if(self.conv_dim == '1d'):
            out = self.encoder(x)
            out = torch.squeeze(out, 2)
            out = out.permute(0, 2, 1)
            h = out
            out = self.attn(out)
            out = h + out
            out = out.permute(1, 0, 2)
            self.lstm.flatten_parameters()
            out, _ = self.lstm(out)
            out = out[-1]
            out = self.fc(out)
        elif(self.conv_dim == '2d'):
            out = self.encoder(x)
            out = out.permute(0, 3, 1, 2)
            h = out
            new_out_shape = out.size()[:2] + (out.size()[2] * out.size()[3],)
            out = out.view(*new_out_shape)
            out = self.attn(out)
            out = out.view(h.size())
            out = h + out
            out = self.gap(out)
            out = torch.squeeze(out, 2)
            out = out.permute(1, 0, 2)
            self.lstm.flatten_parameters()
            out, _ = self.lstm(out)
            out = out[-1]
            out = self.fc(out)
        return out


