

import math
import torch
import numpy as np
import torch.nn as nn

from torch.autograd import Variable
from model.VAE import VAE

import torch.nn.functional as F
def get_attn_pad_mask(seq_q, seq_k, pad_index):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(pad_index).unsqueeze(1)
    pad_attn_mask = torch.as_tensor(pad_attn_mask, dtype=torch.int)
    return pad_attn_mask.expand(batch_size, len_q, len_k)

def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask # [batch_size, tgt_len, tgt_len]

class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_k, device):
        super(ScaledDotProductAttention, self).__init__()
        self.device = device
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        # print(Q)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn_mask = torch.as_tensor(attn_mask, dtype=torch.bool)
        attn_mask = attn_mask.to(self.device)
        # print(attn_mask)
        scores.masked_fill_(attn_mask, -1e9).to(self.device)
        # print(scores)

        attn = nn.Softmax(dim=-1)(scores).to(self.device)
        context = torch.matmul(attn, V)
        return context, attn


def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).int()
    return subsequent_mask


class GELU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, dim, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., dim, 2) *
                             -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)





class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, n_heads,device):
        super(MultiHeadAttention, self).__init__()
        self.WQ = nn.Linear(d_model, d_k * n_heads)

        self.WK = nn.Linear(d_model, d_k * n_heads)
        self.WV = nn.Linear(d_model, d_v * n_heads)

        self.linear = nn.Linear(n_heads * d_v, d_model)


        self.layer_norm = nn.LayerNorm(d_model)
        self.device = device

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.shape[0]
        # print(K.shape)
        # print(V.shape)

        q_s = self.WQ(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # print(K.shape)
        # print(self.WK(K))


        k_s = self.WK(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.WV(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1).to(self.device)
        context, attn = ScaledDotProductAttention(d_k=self.d_k,device=self.device)(
            Q=q_s, K=k_s, V=v_s, attn_mask=attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.linear(context)
        return self.layer_norm(output + Q), attn


class MultiEncoderAttention(nn.Module):

    def __init__(self, d_model, d_k_OP, d_v_OP, n_heads_OP, d_k_TP, d_v_TP, n_heads_TP,device):
        super(MultiEncoderAttention, self).__init__()
        self.WQ_OP = nn.Linear(d_model, d_k_OP * n_heads_OP)
        self.WK_OP = nn.Linear(d_model, d_k_OP * n_heads_OP)
        self.WV_OP = nn.Linear(d_model, d_v_OP* n_heads_OP)

        self.WQ_TP = nn.Linear(d_model, d_k_TP * n_heads_TP)
        self.WK_TP= nn.Linear(d_model, d_k_TP* n_heads_TP)
        self.WV_TP= nn.Linear(d_model, d_v_TP* n_heads_TP)

        self.linear = nn.Linear((d_v_OP* n_heads_OP+d_v_TP* n_heads_TP), d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        self.device = device

        self.d_model = d_model
        self.d_k_OP = d_k_OP
        self.d_v_OP= d_v_OP
        self.n_heads_OP = n_heads_OP

        self.d_k_TP = d_k_TP
        self.d_v_TP = d_v_TP
        self.n_heads_TP = n_heads_TP


    def forward(self, Q_OP, K_OP, V_OP, attn_mask_OP,Q_TP,K_TP,V_TP,attn_mask_TP):
        batch_size = Q_OP.shape[0]
        q_s_OP = self.WQ_OP(Q_OP).view(batch_size, -1, self.n_heads_OP, self.d_k_OP).transpose(1, 2)
        k_s_OP = self.WK_OP(K_OP).view(batch_size, -1, self.n_heads_OP, self.d_k_OP).transpose(1, 2)
        v_s_OP = self.WV_OP(V_OP).view(batch_size, -1, self.n_heads_OP, self.d_v_OP).transpose(1, 2)

        q_s_TP = self.WQ_TP(Q_TP).view(batch_size, -1, self.n_heads_TP, self.d_k_TP).transpose(1, 2)
        k_s_TP = self.WK_TP(K_TP).view(batch_size, -1, self.n_heads_TP, self.d_k_TP).transpose(1, 2)
        v_s_TP = self.WV_TP(V_TP).view(batch_size, -1, self.n_heads_TP, self.d_v_TP).transpose(1, 2)

        attn_mask_OP = attn_mask_OP.unsqueeze(1).repeat(1, self.n_heads_OP, 1, 1)
        context_OP, attn_OP = ScaledDotProductAttention(d_k=self.d_k_OP,device=self.device)(
            Q=q_s_OP, K=k_s_OP, V=v_s_OP, attn_mask=attn_mask_OP)
        context_OP = context_OP.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads_OP * self.d_v_OP)

        attn_mask_TP = attn_mask_TP.unsqueeze(1).repeat(1, self.n_heads_TP, 1, 1)
        context_TP, attn_TP = ScaledDotProductAttention(d_k=self.d_k_TP,device=self.device)(
            Q=q_s_TP, K=k_s_TP, V=v_s_TP, attn_mask=attn_mask_TP,)
        context_TP = context_TP.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads_TP * self.d_v_TP)

        context=torch.cat((context_OP,context_TP),dim=-1)
        output = self.linear(context)



        return self.layer_norm(output + Q_OP), attn_OP,attn_TP


class PoswiseFeedForwardNet(nn.Module):

    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)

        self.relu = GELU()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs
        output = self.l1(inputs)
        output = self.relu(output)
        output = self.l2(output)
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads,device):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads,device=device)
        self.pos_ffn = PoswiseFeedForwardNet(
            d_model=d_model, d_ff=d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(
            Q=enc_inputs, K=enc_inputs,
            V=enc_inputs, attn_mask=enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class Encoder(nn.Module):

    def __init__(self, EmbLayer,d_model, d_ff, d_k, d_v, n_heads, n_layers, pad_index=0,device=None,segLayer=None):
        super(Encoder, self).__init__()
        # self.device = device
        self.pad_index = pad_index
        self.EmbLayer = EmbLayer
        self.segLayaer = segLayer
        self.pos_emb = PositionalEncoding(
            dim=d_model,
            dropout=0.1)
        self.layers = []
        for _ in range(n_layers):
            encoder_layer = EncoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v, n_heads=n_heads,device=device
               )
            self.layers.append(encoder_layer)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x_input, seg_input=None):
        # enc_outputs = self.src_emb(x)
        # print(x_input.shape)
        # print(x_input_emb.shape)


        x_input_emb = self.EmbLayer(x_input)
        enc_outputs = self.pos_emb(x_input_emb)
        if seg_input!= None:
            enc_outputs =enc_outputs + self.segLayaer(seg_input)
        enc_self_attn_mask = get_attn_pad_mask(x_input, x_input, self.pad_index)

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        enc_self_attns = torch.stack(enc_self_attns)
        enc_self_attns = enc_self_attns.permute([1, 0, 2, 3, 4])
        return enc_outputs, enc_self_attns



class Concat_Encoder(nn.Module):

    def __init__(self,d_model, d_ff, d_k, d_v, n_heads, n_layers, pad_index=0,device=None):
        super(Concat_Encoder, self).__init__()
        # self.device = device
        self.pad_index = pad_index

        self.layers = []
        for _ in range(n_layers):
            encoder_layer = EncoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v, n_heads=n_heads,device=device
               )
            self.layers.append(encoder_layer)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x_input,enc_outputs):
        # enc_outputs = self.src_emb(x)
        # print(x_input.shape)
        # print(x_input_emb.shape)

        enc_self_attn_mask = get_attn_pad_mask(x_input, x_input, self.pad_index)

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        enc_self_attns = torch.stack(enc_self_attns)
        enc_self_attns = enc_self_attns.permute([1, 0, 2, 3, 4])
        return enc_outputs, enc_self_attns

class AttriEncoder(nn.Module):

    def __init__(self,input_text_emb,aspect_type_emb,label_emb,user_emb,business_emb,input_aspect_dim,label_dim,
                 user_dim,business_dim,d_model,padding_idx,device,opt):
        super(AttriEncoder, self).__init__()
        text_embed_size = input_text_emb.shape[1]
        self.hidden_dim =100
        self.use_keywords =opt.use_keywords
        self.key_encoder_layers = opt.opinion_encoder_layers
        self.pos_emb = PositionalEncoding(
            dim=d_model,
            dropout=0)


        self.aspect_type_EmbLayer = nn.Embedding.from_pretrained(torch.tensor(aspect_type_emb,
                                                                              dtype=torch.float), freeze=False,
                                                                 padding_idx=padding_idx)

        self.user_EmbLayer = nn.Embedding.from_pretrained(torch.tensor(user_emb,
                                                                       dtype=torch.float), freeze=False,
                                                          padding_idx=padding_idx)

        self.business_EmbLayer = nn.Embedding.from_pretrained(torch.tensor(business_emb,
                                                                           dtype=torch.float), freeze=False,
                                                              padding_idx=padding_idx)

        self.label_EmbLayer = nn.Embedding.from_pretrained(torch.tensor(label_emb,
                                                                        dtype=torch.float), freeze=False,
                                                           padding_idx=padding_idx)
        if self.use_keywords =="yes":
            self.key_words_EmbLayer = nn.Embedding.from_pretrained(torch.tensor(input_text_emb,
                                                                        dtype=torch.float), freeze=False,
                                                           padding_idx=padding_idx)
            # print(opt)
            self.key_encoder = Concat_Encoder(text_embed_size,opt.opinion_dim_feedforward,opt.K_V_dim,opt.K_V_dim,opt.n_heads,
                                       opt.opinion_encoder_layers, pad_index=padding_idx,device=device)
            self.out_encoder_liner =nn.Linear(in_features=d_model+self.hidden_dim+label_dim,out_features=d_model)
        self.other_MLP = nn.Linear(in_features=(input_aspect_dim+business_dim+user_dim),out_features=self.hidden_dim,)
        self.user_bus_MLP =nn.Linear(in_features=(business_dim+user_dim),out_features=self.hidden_dim)
        # self.VAE =VAE(d_model,self.hidden_dim,z_dim=30)

    def forward(self, key_words,aspect_type,label,user_id,business_id):
        """

        :param aspect_type:  输入的aspect类型 batch*1*index
        :param label:  输入的情感标签   batch*1*index
        :param user_id:  用户的信息 batch*1*emb*index
        :param business_id:  product的信息 batch*1*index
        :return:    encoder 的hidden batch*1*emb*index
        """


        # print(aspect_type)
        aspect_type_emb = self.aspect_type_EmbLayer(aspect_type).squeeze(1)
        label_emb = self.label_EmbLayer(label).squeeze(1)
        user_emb = self.user_EmbLayer(user_id)
        business_emb = self.business_EmbLayer(business_id)
        key_emb = self.key_words_EmbLayer(key_words)

        # print(aspect_type_emb.shape)
        # print(user_emb.shape)
        # print(business_emb.shape)
        other_input = torch.cat((aspect_type_emb,user_emb,business_emb),dim=-1)

        # print(aspect_input.shape)

        # print(user_bus.shape)
        other_hidden = torch.tanh(self.other_MLP(other_input)).squeeze(1)
        # print(aspect_hidden.shape)
        # aspect_hidden = F.relu(aspect_hidden).squeeze()
        # other_hidden  =F.relu(other_hidden)

        # print(other_hidden.shape)
        # print(aspect_hidden)

        attru_outpus = torch.cat((label_emb,other_hidden),dim=-1)
        attru_outpus = attru_outpus.unsqueeze(1)
        # print(key_words.shape)
        # print(key_words.shape[1])
        attru_outpus = attru_outpus.repeat(1, key_words.shape[1], 1)
        attru_outpus = torch.cat((key_emb, attru_outpus), dim=-1)
        attru_outpus = self.out_encoder_liner(attru_outpus)

        if self.use_keywords =="yes":
            attru_outpus, _ = self.key_encoder(key_words,attru_outpus)

        return attru_outpus



class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads,device):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads,device=device)
        self.dec_enc_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads,device=device)
        self.pos_ffn = PoswiseFeedForwardNet(
            d_model=d_model, d_ff=d_ff)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # print(enc_outputs.shape)
        # exit()
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

class Fusion_DecoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads,device=None):
        super(Fusion_DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads,device=device)
        self.dec_enc_attn = MultiEncoderAttention(d_model=d_model,d_k_OP=d_k,
                                                  d_v_OP=d_v, n_heads_OP=n_heads,
                                                  d_k_TP=d_k, d_v_TP=d_v, n_heads_TP=n_heads,device=device)
        self.pos_ffn = PoswiseFeedForwardNet(
            d_model=d_model, d_ff=d_ff)

    def forward(self, dec_inputs, review_enc_outputs,template_enc_outputs, dec_self_attn_mask, dec_enc_attn_mask_OP,dec_enc_attn_mask_TP):
        # dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask
        # Q, K, V, attn_mask
        # Q_OP, K_OP, V_OP, attn_mask_OP, Q_TP, K_TP, V_TP, attn_mask_TP
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, attn_OP,attn_TP = self.dec_enc_attn(dec_outputs, review_enc_outputs,review_enc_outputs,dec_enc_attn_mask_OP, dec_outputs,template_enc_outputs,template_enc_outputs,dec_enc_attn_mask_TP)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, attn_OP,attn_TP

class Decoder(nn.Module):

    def __init__(self,passage_text_emb_EmbLayer,out_voc_size, d_model, d_ff, d_k, d_v, n_heads, n_layers, pad_index,device):
        super(Decoder, self).__init__()
        self.pad_index = pad_index
        self.device = device
        # self.tgt_emb = outputs_emb
        self.tgt_emb = passage_text_emb_EmbLayer
        self.device = device
        self.pos_emb = PositionalEncoding(
            dim=d_model,
            dropout=0)
        self.layers = []
        for _ in range(n_layers):
            decoder_layer = DecoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v,
                n_heads=n_heads,device=device )
            self.layers.append(decoder_layer)
        self.layers = nn.ModuleList(self.layers)


    def forward(self,dec_inputs, enc_inputs, enc_outputs):

        dec_outputs = self.tgt_emb(dec_inputs)

        dec_outputs = self.pos_emb(dec_outputs)



        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.pad_index)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs).to(device=self.device)
        # print(dec_self_attn_pad_mask.device)
        # print(dec_self_attn_subsequent_mask.device)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs, self.pad_index)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(
                dec_inputs=dec_outputs,
                enc_outputs=enc_outputs,
                dec_self_attn_mask=dec_self_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        dec_self_attns = torch.stack(dec_self_attns)
        dec_enc_attns = torch.stack(dec_enc_attns)

        dec_self_attns = dec_self_attns.permute([1, 0, 2, 3, 4])
        dec_enc_attns = dec_enc_attns.permute([1, 0, 2, 3, 4])

        return dec_outputs, dec_self_attns, dec_enc_attns



class Fusion_Decoder(nn.Module):

    def __init__(self,pos_EmbLayer,review_embed,d_model,fusion_decoder_layers,d_ff,d_k,d_v,n_heads,padding_idx,device):
        super(Fusion_Decoder, self).__init__()
        self.pad_index =padding_idx
        self.d_model = d_model
        # self.device = device
        self.review_EmbLayer = nn.Embedding.from_pretrained(torch.tensor(review_embed,
                                                                         dtype=torch.float), freeze=False,
                                                            padding_idx=padding_idx)
        self.pos_EmbLayer = pos_EmbLayer

        self.pos_emb = PositionalEncoding(
            dim=d_model,
            dropout=0)
        self.decoder_layers = []
        self.device =device
        for _ in range(fusion_decoder_layers):
            decoder_layer = Fusion_DecoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v,
                n_heads=n_heads,device=device)
            self.decoder_layers.append(decoder_layer)
        self.decoder_layers = nn.ModuleList(self.decoder_layers)

    def forward(self, dec_inputs,dec_inputs_pos, enc_inputs_OP, enc_outputs_OP,enc_inputs_TP, enc_outputs_TP):
        dec_inputs_emb = self.review_EmbLayer(dec_inputs)
        dec_inputs_emb = self.pos_emb(dec_inputs_emb)
        # dec_inputs_emb = dec_inputs_emb + self.pos_EmbLayer(dec_inputs_pos)
        dec_outputs = dec_inputs_emb
        # print(dec_inputs)


        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.pad_index)
        # print(dec_self_attn_pad_mask)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs).to(self.device)
        # print(dec_self_attn_subsequent_mask )
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        # print(dec_self_attn_mask)
        # exit()
        dec_enc_attn_mask_OP = get_attn_pad_mask(dec_inputs, enc_inputs_OP, self.pad_index)

        dec_enc_attn_mask_TP = get_attn_pad_mask(dec_inputs, enc_inputs_TP, self.pad_index)


        dec_self_attns, dec_enc_attns_OP,dec_enc_attns_TP = [], [],[]
        for layer in self.decoder_layers:
            dec_outputs, dec_self_attn, attn_OP, attn_TP = layer(dec_inputs=dec_outputs, review_enc_outputs=enc_outputs_OP,
                                                             template_enc_outputs=enc_outputs_TP, dec_self_attn_mask=dec_self_attn_mask,
                                                             dec_enc_attn_mask_OP=dec_enc_attn_mask_OP,dec_enc_attn_mask_TP=dec_enc_attn_mask_TP)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns_OP.append(attn_OP)
            dec_enc_attns_TP.append(attn_TP)
        dec_self_attns = torch.stack(dec_self_attns)
        ec_enc_attns_OP= torch.stack(dec_enc_attns_OP)
        dec_enc_attns_TP = torch.stack(dec_enc_attns_TP)

        dec_self_attns = dec_self_attns.permute([1, 0, 2, 3, 4])
        dec_enc_attns_OP= ec_enc_attns_OP.permute([1, 0, 2, 3, 4])
        dec_enc_attns_TP = dec_enc_attns_TP.permute([1, 0, 2, 3, 4])

        return dec_outputs, dec_enc_attns_OP, dec_enc_attns_TP




class MaskedDecoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, device):
        super(MaskedDecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads, device=device)
        self.pos_ffn = PoswiseFeedForwardNet(
            d_model=d_model, d_ff=d_ff)

    def forward(self, dec_inputs, dec_self_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn


class MaskedDecoder(nn.Module):

    def __init__(self, vocab_size, d_model, d_ff, d_k,
                 d_v, n_heads, n_layers, pad_index, device):
        super(MaskedDecoder, self).__init__()
        self.pad_index = pad_index
        self.tgt_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(
            dim=d_model,
            dropout=0)

        self.layers = []
        for _ in range(n_layers):
            decoder_layer = MaskedDecoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v, n_heads=n_heads,
                device=device)
            self.layers.append(decoder_layer)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, dec_inputs):
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs)

        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.pad_index)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_self_attns = []
        for layer in self.layers:
            dec_outputs, dec_self_attn = layer(
                dec_inputs=dec_outputs,
                dec_self_attn_mask=dec_self_attn_mask)
            dec_self_attns.append(dec_self_attn)
        dec_self_attns = torch.stack(dec_self_attns)
        dec_self_attns = dec_self_attns.permute([1, 0, 2, 3, 4])
        return dec_outputs, dec_self_attns

class Opinion_Transformer(nn.Module):
    def __init__(self,input_text_emb,aspect_type_emb,label_emb,user_emb,business_emb,passage_text_emb,passage_text_emb_EmbLayer,
                 K_V_dim,dim_feedforward,n_heads,n_layers,padding_idx,device,opt):
        super(Opinion_Transformer, self).__init__()


        self.passage_text_emb_EmbLayer = passage_text_emb_EmbLayer


        self.pad_idx = padding_idx
        passage_voc_size = passage_text_emb.shape[0]
        d_model = input_text_emb.shape[1]
        aspect_type_dim = aspect_type_emb.shape[1]
        label_dim = label_emb.shape[1]
        user_dim = user_emb.shape[1]
        business_dim = business_emb.shape[1]
        self.K_V_dim = K_V_dim
        self.attribute_encoder = None
        self.use_keywords = opt.use_keywords

        self.encoder = AttriEncoder(input_text_emb,aspect_type_emb, label_emb, user_emb,
                                        business_emb, aspect_type_dim, label_dim, user_dim, business_dim,
                                        d_model=d_model, padding_idx=padding_idx,device = device,opt=opt)

        self.decoder =  Decoder(self.passage_text_emb_EmbLayer,passage_voc_size,d_model,dim_feedforward,K_V_dim,K_V_dim,
                                n_heads,n_layers=n_layers,pad_index=padding_idx,device = device)
        self.vae = VAE(h_dim=d_model,z_dim=d_model)


    def forward(self,input_text,aspect_type,label,user_id,business_id,passage_list):
        # print(input_text.shape)
        # exit()
        aspect_type = aspect_type.unsqueeze(-1)
        label =label.unsqueeze(-1)
        # print(aspect_type.shape)
        # print(label.shape)
        # exit()

        # print(passage_list.dtype)

        # print(passage_emb.shape)
        # print("-"*30)
        # if
        if self.use_keywords =="yes":
            enc_inputs = input_text
        else:
            enc_inputs = aspect_type
        enc_outputs = self.encoder(input_text,aspect_type,label,user_id,business_id)  #encoder_outputs B*1*emb'
        #enc_outputs_reconstruct = self.vae(enc_outputs)
        opinion_dec_outputs, opinion_dec_self_attns, opinion_dec_enc_attns = self.decoder(passage_list, enc_inputs, enc_outputs)
        #KL_reconstrcut = F.kl_div(enc_outputs_reconstruct.softmax(dim=-1).log(), enc_outputs.softmax(dim=-1), reduction='sum')
        KL_reconstrcut = 0
        # KL_reconstrcut_re = F.kl_div(enc_outputs.softmax(dim=-1).log(), enc_outputs_reconstrcut.softmax(dim=-1), reduction='mean')
        return  opinion_dec_outputs, opinion_dec_self_attns, opinion_dec_enc_attns,KL_reconstrcut




class Fusion_Transformer(nn.Module):
    def __init__(self,passage_EmbLayer,pos_embed ,review_embed,
                review_encoder_layers,template_encoder_layers,
                 fusion_decoder_layers,dim_feedforward,n_heads,K_V_dim,padding_idx,device):
        super(Fusion_Transformer, self).__init__()
        self.passage_EmbLayer=passage_EmbLayer
        self.review_voc_size, text_emb_dim = review_embed.shape
        self.pos_EmbLayer = nn.Embedding.from_pretrained(torch.tensor(pos_embed,
                                                                          dtype=torch.float), freeze=True,
                                                             padding_idx=padding_idx)

        self.sge_EmbLayer = nn.Embedding(20, text_emb_dim, padding_idx=0)
        self.device = device

        pos_dim = pos_embed.shape[1]
        padding_idx  =0
        self.opinion_enc = Encoder(self.passage_EmbLayer,text_emb_dim,dim_feedforward,K_V_dim,K_V_dim,n_heads,review_encoder_layers, pad_index=padding_idx,device=device,segLayer=self.sge_EmbLayer)
        self.template_enc = Encoder(self.pos_EmbLayer,pos_dim,dim_feedforward,K_V_dim,K_V_dim,n_heads,template_encoder_layers, pad_index=padding_idx,device=device)
        print("----------------------------------this is Fusion_Decoder-----------------------------")
        self.fusion_decoder = Fusion_Decoder(self.pos_EmbLayer,review_embed,text_emb_dim,fusion_decoder_layers,dim_feedforward,K_V_dim,K_V_dim,n_heads,padding_idx,device=device)

    def forward(self,enc_inputs_OP,enc_inputs_TP,dec_inputs,dec_inputs_pos,segment_input):
        # print(enc_inputs_OP.shape)
        enc_outputs_OP, enc_self_attns_OP = self.opinion_enc(enc_inputs_OP,segment_input)
        enc_outputs_TP, enc_self_attns_TP = self.template_enc(enc_inputs_TP)




        dec_outputs, ec_enc_attns_OP, ec_enc_attns_TP = self.fusion_decoder(dec_inputs, dec_inputs_pos,enc_inputs_OP, enc_outputs_OP,enc_inputs_TP, enc_outputs_TP)
        return dec_outputs, ec_enc_attns_OP, ec_enc_attns_TP

