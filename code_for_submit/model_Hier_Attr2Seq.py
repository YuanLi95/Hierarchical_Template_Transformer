

import math
from typing import List

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from beam_search import BeamSearch, NgramBlocking
from sumeval.metrics.bleu import BLEUCalculator
from sumeval.metrics.rouge import RougeCalculator
# from torchtext.vocab import Vocab
from other_model.Hei_Attr2seq import Heir_Att2Seq,Review_Base_Transformer
# from model.transformer import Opinion_Transformer,Fusion_Transformer
from torch.nn.utils.rnn import pad_sequence
import copy
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_ppl import  ppl_score
smooth = SmoothingFunction()


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self,
                 label_smoothing: float,
                 tgt_vocab_size: int,
                 device: torch.device,
                 ignore_index: int = -100,
                 reduction: str = "none"):
        assert 0.0 < label_smoothing <= 1.0
        assert reduction in ["sum", "mean", "batchmean", "none"]
        self.device = device
        self.ignore_index = ignore_index
        self.reduction = reduction
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0

        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = (1.0 - label_smoothing)

    def forward(self,
                output: torch.FloatTensor,
                target: torch.LongTensor):
        """

        """
        output = output.transpose(0, 1)
        target = target.transpose(0, 1)
        model_prob = self.one_hot.repeat(target.size(0), target.size(1), 1).to(self.device)
        model_prob.scatter_(2, target.unsqueeze(2), self.confidence)

        model_prob.masked_fill((target == self.ignore_index).unsqueeze(2), 0)
        # print(F.kl_div(F.log_softmax(output,dim=-1),
        #                     model_prob,
        #                     reduction="none").transpose(0,1).sum(2))
        # print(F.kl_div(F.log_softmax(output,dim=-1),
        #                     model_prob,
        #                     reduction="none").transpose(0,1).sum(2).shape)
        if self.reduction == "none":
            return F.kl_div(F.log_softmax(output, dim=-1),
                            model_prob,
                            reduction="none").transpose(0, 1).sum(2)
        else:
            return F.kl_div(F.log_softmax(output, dim=-1),
                            model_prob,
                            reduction=self.reduction).transpose(0, 1)


class PositionalEncoding(nn.Module):
    """OpenNMT-py"""
    """Sinusoidal positional encoding for non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dim, dropout=0.1, max_len=5000):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        """Embed inputs.

        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """

        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.dropout(emb)
        return emb


class Hier_Attr2Seq_model(nn.Module):
    def __init__(self,
                 input_text_embedding,
                 passage_list_embedding,
                 out_text_embedding,
                 user_embedding,
                 business_embedding,
                 label_embedding,
                 aspect_type_embedding,
                 pos_embedding,
                 n_heads,
                 opinion_encoder_layers,
                 opinion_decoder_layers,
                 review_encoder_layers,
                 template_encoder_layers,
                 fusion_decoder_layers,
                 dim_feedforward,
                 dropout,
                 padding_idx,
                 K_V_dim=512,
                 device=None,
                 opt=None

                 ):
        padding_idx = padding_idx
        super(Hier_Attr2Seq_model, self).__init__()
        self.text_dim = input_text_embedding.shape[1]

        self.user_dim = user_embedding.shape[1]
        self.business_dim = business_embedding.shape[1]

        self.opinion_voc_size = passage_list_embedding.shape[0]
        self.review_voc_size = out_text_embedding.shape[0]
        self.K_V_dim = K_V_dim
        self.d_model = self.text_dim
        self.device = device
        self.pad_idx = padding_idx
        self.eos_idx = 2
        self.bos_idx = 3
        self.sep_idx = 4
        self.unk_indx = 1
        self.use_keywords = opt.use_keywords

        # self.Key_word_EmbLayer =  nn.Embedding.from_pretrained(torch.tensor(input_text_embedding,
        #                                                                   dtype=torch.float), freeze=True,
        #                                                      padding_idx=padding_idx)
        self.passage_EmbLayer = nn.Embedding.from_pretrained(torch.tensor(passage_list_embedding,
                                                                          dtype=torch.float), freeze=False,
                                                             padding_idx=padding_idx)

        # 需要传入text_embedding和position_embeding 输入和输出的

        self.Heir_Att2Seq = Heir_Att2Seq(input_text_embedding, user_embedding, business_embedding,label_embedding,
                                           passage_list_embedding,padding_idx=padding_idx, device=device,
                                           opt=opt)

        self.opinion_projection = nn.Linear(self.d_model,
                                            self.opinion_voc_size)

        self.review_Fusionsformer = Review_Base_Transformer(self.passage_EmbLayer, pos_embedding
                                                       , out_text_embedding,
                                                       review_encoder_layers,
                                                       template_encoder_layers,
                                                       fusion_decoder_layers,
                                                       dim_feedforward,
                                                       n_heads,
                                                       self.K_V_dim,
                                                       padding_idx=padding_idx,
                                                       device=device
                                                       )
        self.review_projection = nn.Linear(self.d_model,
                                           self.review_voc_size)

    def forward(self, input_text, aspect_type, label, user_id, business_id, passage_list, temp_pos, review_output,review_pos):
        opinion_outputs_list = []
        # print(aspect_type.shape)
        # print(passage_list.shape)
        # exit()
        # 有多个aspect重构
        bacth_size, aspect_num = aspect_type.shape

        opinion_outputs_argmax = []
        # review_inputs = torch.tensor()
        review_bos = [torch.tensor([self.bos_idx]).to(device=self.device)] * bacth_size
        # print(review_bos)
        KL_loss_reconstrcut = []
        segment_input = [torch.tensor([1]).to(device=self.device)] * bacth_size
        for i in range(aspect_num):
            # print(i)
            opinion_outputs =self.Heir_Att2Seq(
                input_text[:, i, :],  label[:, i], user_id, business_id, passage_list[:, i, :])
            opinion_outputs = self.opinion_projection(opinion_outputs)

            # print(opinion_outputs.shape)
            opinion_outputs_list.append(opinion_outputs)

            opinion_out_i = torch.argmax(opinion_outputs, dim=-1)
            # print(opinion_out_i)

            # print(opinion_out_i)

            opinion_out_i_pad = opinion_out_i != self.pad_idx
            opinion_out_i_eos = opinion_out_i != self.eos_idx
            opinion_out_i_bos = opinion_out_i != self.bos_idx
            opinion_out_i_unk = opinion_out_i != self.unk_indx
            opinion_need_put_idx = opinion_out_i_pad.mul(opinion_out_i_eos)
            opinion_need_put_idx = opinion_need_put_idx.mul(opinion_out_i_bos)
            opinion_need_put_idx = opinion_need_put_idx.mul(opinion_out_i_unk)

            for j in range(bacth_size):
                # 第i个aspect的第J个bacth
                if  (label[j, i])==self.pad_idx:
                    continue
                opinion_need_put_i_j = opinion_out_i[j, :][opinion_need_put_idx[j, :]]
                review_bos[j] = torch.cat((review_bos[j], opinion_need_put_i_j))
                if i != (aspect_num - 1):
                    if label[j, i+1]!=self.pad_idx:
                        review_bos[j] = torch.cat((review_bos[j], torch.tensor([self.sep_idx]).long().to(self.device)))
                    else:
                        review_bos[j] = torch.cat((review_bos[j], torch.tensor([self.eos_idx]).long().to(self.device)))

                else:
                    review_bos[j] = torch.cat((review_bos[j], torch.tensor([self.eos_idx]).long().to(self.device)))
                segment_input[j] = torch.cat((segment_input[j], torch.tensor([i+1]*(len(opinion_need_put_i_j)+1)).long().to(self.device)))
        # print(review_bos)
        review_input = pad_sequence(review_bos, padding_value=0, batch_first=True)
        segment_input = pad_sequence(segment_input, padding_value=0, batch_first=True)


        # print(review_input.shape)
        # print(review_input)

        dec_outputs, dec_self_attns, dec_enc_attns = self.review_Fusionsformer(review_input, temp_pos, review_output,review_pos,segment_input)

        review_outputs = self.review_projection(dec_outputs)
        # print(review_outputs.shape)
        return opinion_outputs_list, review_outputs, review_input

    def opinion_encode(self, input_text, aspect_type, aspect_label, user, business):
        memory = self.opinion_Transformer.encoder(input_text, aspect_type, aspect_label, user, business)
        return memory

    def opinion_decode(self, tgt, enc_inputs, memory, ):
        output = self.opinion_Transformer.decoder(tgt, enc_inputs, memory)
        return output

    def opinion_generate(self, input_text, aspect_type, aspect_label, user, business, maxlen, bos_index, pad_index,
                         opt, ):
        # Obtain device information
        device = next(self.parameters()).device
        batch_size, seq_len = input_text.shape
        attr_memory = []

        memory_i,user_input,business_input,rating_input = self.Heir_Att2Seq.Att_encoder(input_text,  user, business,aspect_label)

        # <BOS> tgt seq for generation
        tgt = torch.LongTensor(batch_size, maxlen).fill_(pad_index).to(device)
        tgt[:, 0] = torch.LongTensor(batch_size).fill_(bos_index).to(device)
        for i in range(1, maxlen):
            decode_prob =self.Heir_Att2Seq.decoder(tgt[:, :i], memory_i,user_input,business_input,rating_input)
            pred_prob = self.opinion_projection(decode_prob)
            decode_output = pred_prob.argmax(-1)
            tgt[:, i] = decode_output[:, -1]
        return tgt

    # review_input, temp_pos, maxlen, bos_index, pad_index, opt,
    def review_generate(self, opinion_input, temp_input, maxlen, bos_index, pad_index, opt, ):

        batch_size, input_seq_len = opinion_input.shape
        enc_outputs_OP, enc_self_attns_OP = self.review_Fusionsformer.opinion_enc(opinion_input)
        enc_outputs_TP, enc_self_attns_TP = self.review_Fusionsformer.template_enc(temp_input)

        tgt = torch.LongTensor(batch_size, maxlen).fill_(pad_index).to(self.device)
        tgt[:, 0] = torch.LongTensor(batch_size).fill_(bos_index).to(self.device)

        for i in range(1, maxlen):
            # dec_inputs, enc_inputs_OP, enc_outputs_OP, enc_inputs_TP, enc_outputs_TP


            decode_prob, _1, _2 = self.review_Fusionsformer.fusion_decoder(tgt[:, :i], opinion_input, enc_outputs_OP,
                                                                           temp_input, enc_outputs_TP)
            pred_prob = self.review_projection(decode_prob)
            decode_output = pred_prob.argmax(-1)
            tgt[:, i] = decode_output[:, -1]
        return tgt

    def review_generate_beam_search(self, opinion_input, temp_input,out2pos, maxlen, bos_index, pad_index, opt,
                                    vocab_size, beam_size, no_repeat_ngram_size=0, ):
        """

        :param opinion_input:  Batch_s_e
        :param temp_input:  b_s_e
        :param maxlen:
        :param bos_index:
        :param pad_index:
        :param opt:
        :param vocab_size: tgt_voc_size
        :param beam_size:
        :param no_repeat_ngram_size:
        :return:
        """
        # print(opinion_input.shape)

        batch_size, input_seq_len = opinion_input.shape
        enc_outputs_OP, enc_self_attns_OP = self.review_Fusionsformer.opinion_enc(opinion_input)
        tgt = torch.LongTensor(batch_size, maxlen, beam_size).fill_(self.pad_idx).to(self.device)
        tgt[:, 0, :] = torch.LongTensor(batch_size, beam_size).fill_(self.bos_idx).to(self.device)
        # tgt.shape b*max_len*beam_size

        scores = torch.zeros(batch_size, beam_size, maxlen).to(self.device)
        scores[:, :, 0] = torch.ones(batch_size, beam_size).to(self.device)
        active_beams = [0]  # up to beam_size beams.
        search = BeamSearch(vocab_size, self.pad_idx, self.unk_indx, self.eos_idx)
        ngram_blocking = NgramBlocking(no_repeat_ngram_size)

        # After eos
        log_probs_after_eos = torch.FloatTensor(batch_size, beam_size, vocab_size).fill_(float("-inf")).cpu()
        log_probs_after_eos[:, :, self.eos_idx] = 0.
        best_n_indices = tgt.new_full((batch_size, len(active_beams)), bos_index)

        for i in range(1, maxlen):
            if (best_n_indices == self.eos_idx).all():  # if all of last prediction is eos, we can leave the loop
                break

            # Generate probability for all beams, update probability for all beams (lprobs).
            lprobs = torch.zeros(batch_size, len(active_beams), vocab_size).to(self.device)
            for j in range(len(active_beams)):
                tgt_pos = temp_input[:, :i]
                decode_prob, _1, _2 = self.review_Fusionsformer.fusion_decoder(tgt[:, :i, active_beams[j]],
                                                                               opinion_input, enc_outputs_OP)

                #base
                # decode_prob, _1, _2 = self.review_Fusionsformer.fusion_decoder(tgt[:, :i, active_beams[j]],opinion_input, enc_outputs_OP, )
                pred_prob = self.review_projection(decode_prob)

                lprobs[:, j, :] = pred_prob[:, -1, :]

            # Update lprobs for n-gram blocking
            if no_repeat_ngram_size > 0:
                for batch_idx in range(batch_size):
                    for beam_idx in range(len(active_beams)):
                        lprobs[batch_idx, beam_idx] = ngram_blocking.update(i - 1,
                                                                            tgt[batch_idx, :i, beam_idx],
                                                                            lprobs[batch_idx, beam_idx])

            expanded_indices = best_n_indices.detach().cpu().unsqueeze(-1).expand(
                (batch_size, len(active_beams), vocab_size))
            clean_lprobs = torch.where(expanded_indices == self.eos_idx, log_probs_after_eos[:, :len(active_beams)],
                                       F.log_softmax(lprobs.detach().cpu(), dim=-1))
            # Run the beam search step and select the top-k beams.
            best_n_scores, best_n_indices, best_n_beams = search.step(i, clean_lprobs,
                                                                      scores.index_select(1, torch.tensor(active_beams,
                                                                                                          device=self.device)).detach().cpu(),
                                                                      beam_size)

            # Take the top results, more optimization can be done here, e.g., avoid <eos> beams.
            best_n_scores = best_n_scores[:, :beam_size]
            best_n_indices = best_n_indices[:, :beam_size]
            best_n_beams = best_n_beams[:, :beam_size]

            # update results

            tgt = tgt.transpose(0, 1)

            tgt = tgt.gather(2, best_n_beams.unsqueeze(0).expand(maxlen, batch_size, -1).to(self.device))
            tgt = tgt.transpose(0, 1)
            tgt[:, i, :] = best_n_indices
            scores[:, :, i] = best_n_scores
            active_beams = range(beam_size)

        return tgt[:, :, 0]




    def generate_all_beamsearch(self, input_text, aspect_type,aspect_label,user, business, temp_pos, review_pos, opinion_max_len,
                                review_max_len, bos_index, pad_index, opt, vocab_size, beam_size=3,
                                no_repeat_ngram_size=0):
        opinion_outputs_list = []
        bacth_size, aspect_num = aspect_type.shape
        opinion_outputs_argmax = []
        # print("#"*50)
        review_bos = [torch.tensor([self.bos_idx]).to(device=self.device)] * bacth_size
        segment_input = [torch.tensor([1]).to(device=self.device)] * bacth_size

        for i in range(aspect_num):
            # print(user)
            opinion_tgt = self.opinion_generate(input_text[:, i, :], aspect_type[:, i], aspect_label[:, i], user,
                                                business, opinion_max_len, bos_index, pad_index, opt, )
            opinion_outputs_list.append(opinion_tgt)
            full_pad = torch.full((bacth_size, 1), fill_value=self.sep_idx, ).long().to(self.device)

            opinion_out_i = torch.cat((opinion_tgt, full_pad), dim=-1)
            opinion_out_i_pad = opinion_out_i != self.pad_idx
            opinion_out_i_eos = opinion_out_i != self.eos_idx
            opinion_out_i_bos = opinion_out_i != self.bos_idx
            opinion_out_i_unk = opinion_out_i != self.unk_indx
            opinion_need_put_idx = opinion_out_i_pad.mul(opinion_out_i_eos)
            opinion_need_put_idx = opinion_need_put_idx.mul(opinion_out_i_bos)
            opinion_need_put_idx = opinion_need_put_idx.mul(opinion_out_i_unk)
            for j in range(bacth_size):
                # 第i个aspect的第J个bacth
                if (aspect_label[j, i]) == self.pad_idx:
                    continue
                opinion_need_put_i_j = opinion_out_i[j, :][opinion_need_put_idx[j, :]]
                review_bos[j] = torch.cat((review_bos[j], opinion_need_put_i_j))
                if i != (aspect_num - 1):
                    if aspect_label[j, i + 1] != self.pad_idx:
                        review_bos[j] = torch.cat((review_bos[j], torch.tensor([self.sep_idx]).long().to(self.device)))
                    else:
                        review_bos[j] = torch.cat((review_bos[j], torch.tensor([self.eos_idx]).long().to(self.device)))

                else:
                    review_bos[j] = torch.cat((review_bos[j], torch.tensor([self.eos_idx]).long().to(self.device)))
                segment_input[j] = torch.cat((segment_input[j], torch.tensor([i + 1] * (len(opinion_need_put_i_j) + 1)).long().to(self.device)))

        review_input = pad_sequence(review_bos, padding_value=self.pad_idx, batch_first=True)
        segment_input= pad_sequence(segment_input, padding_value=self.pad_idx, batch_first=True)
        batch_size, temp_pos_len = temp_pos.shape
        maxlen = min(temp_pos_len, review_max_len)
        review_tgt = self.review_generate_beam_search(review_input, temp_pos,review_pos, maxlen, bos_index, pad_index, opt,
                                                      vocab_size,
                                                      beam_size, no_repeat_ngram_size)

        # print(review_tgt.shape)
        # print(review_input.shape)

        return opinion_outputs_list, review_tgt, review_input


def passage_list(id_tensor: torch.Tensor,
                 label2word,
                 ignore_pad: str = "<pad>",
                 ignore_bos: str = "<bos>",
                 ignore_eos: str = "<eos>",
                 ignore_unk: str = "<unk>",
                 join: str = None):
    denum_sentences = []
    sentences = id_tensor.t().tolist()
    # exit()
    for sentence in sentences:
        denum_sentence = []
        for label_id in sentence:
            # print(label_id)
            if ignore_eos == label2word[label_id]:
                denum_sentence.append(label2word[label_id])
                break
            elif ignore_pad == label2word[label_id]:
                break
            denum_sentence.append(label2word[label_id])

        if join is None:
            denum_sentences.append(" ".join(denum_sentence))
        else:
            denum_sentences.append(join.join(denum_sentence))

    return denum_sentences


def type_label(id_tensor: torch.Tensor,
               label2word,
               ignore_pad: str = "<pad>",
               ignore_bos: str = "<bos>",
               ignore_eos: str = "<eos>",
               ignore_unk: str = "<unk>",
               join: str = None):
    denum_sentences = []
    sentences = id_tensor.tolist()
    # exit()
    for sentence in sentences:
        denum_sentence = []
        for label_id in sentence:
            # if ignore_bos == label2word[label_id]:
            #     continue
            # if ignore_eos == label2word[label_id]:
            #     denum_sentence.append(label2word[label_id])
            #     break
            # elif ignore_pad == label2word[label_id]:
            #     break
            denum_sentence.append(label2word[label_id])

        if join is None:
            denum_sentences.append(" ".join(denum_sentence))
        else:
            denum_sentences.append(join.join(denum_sentence))

    return denum_sentences


def denumericalize(id_tensor: torch.Tensor,
                   index2word,
                   ignore_pad: str = "<pad>",
                   ignore_bos: str = "<bos>",
                   ignore_eos: str = "<eos>",
                   ignore_unk: str = "<unk>",
                   join: str = None):
    sentences = id_tensor.tolist()
    denum_sentences = []
    # print(index2word)
    # exit()
    for sentence in sentences:
        denum_sentence = []
        for token_id in sentence:
            if ignore_eos == index2word[token_id]:
                break
            elif ignore_pad == index2word[token_id]:
                break
            elif ignore_bos == index2word[token_id]:
                continue
            else:
                denum_sentence.append(index2word[token_id])

        if join is None:
            denum_sentences.append(" ".join(denum_sentence))
        else:
            denum_sentences.append(join.join(denum_sentence))

    return denum_sentences


class SumEvaluator:
    """Evaluator class for generation.
    A wrapper class of sumeval library
    """

    def __init__(self,
                 metrics: List[str] = ["rouge_1",
                                       "rouge_2",
                                       "rouge_l",
                                       "bleu1",
                                       "bleu2",
                                       "bleu3",
                                       "bleu4",
                                       "bert_ppl"
                                       ],
                 lang: str = "en",
                 stopwords: bool = True,
                 stemming: bool = True,
                 use_porter=True):
        if use_porter:
            self.rouge = RougeCalculator(stopwords=stopwords,
                                         stemming=stemming,
                                         lang="en-porter")
        else:
            self.rouge = RougeCalculator(stopwords=stopwords,
                                         stemming=stemming,
                                         lang="en")
        self.bleu = BLEUCalculator(lang=lang)
        self.metrics = sorted(metrics)

    def eval(self,
             true_gens: List[str],
             pred_gens: List[str]):

        assert len(true_gens) == len(pred_gens)

        eval_list = []
        colnames = []
        for i, (true_gen, pred_gen) in enumerate(zip(true_gens, pred_gens)):
            evals = []

            reference = [true_gen.split()]
            candidate = pred_gen.split()
            # BLEU
            if "bleu1" in self.metrics:
                bleu_score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0),
                                           smoothing_function=smooth.method2)
                evals.append(bleu_score)
            if "bleu2" in self.metrics:
                bleu_score = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0),
                                           smoothing_function=smooth.method2)
                evals.append(bleu_score)
            if "bleu3" in self.metrics:
                bleu_score = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0),
                                           smoothing_function=smooth.method2)
                evals.append(bleu_score)
            if "bleu4" in self.metrics:
                bleu_score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25),
                                           smoothing_function=smooth.method2)
                evals.append(bleu_score)

            # ROUGE
            if "rouge_1" in self.metrics:
                rouge_1 = self.rouge.rouge_n(
                    summary=pred_gen,
                    references=[true_gen],
                    n=1)
                evals.append(rouge_1)

            if "rouge_2" in self.metrics:
                rouge_2 = self.rouge.rouge_n(
                    summary=pred_gen,
                    references=[true_gen],
                    n=2)
                evals.append(rouge_2)

            if "rouge_be" in self.metrics:
                rouge_be = self.rouge.rouge_be(
                    summary=pred_gen,
                    references=[true_gen])
                evals.append(rouge_be)

            if "rouge_l" in self.metrics:
                rouge_l = self.rouge.rouge_l(
                    summary=pred_gen,
                    references=[true_gen])
                evals.append(rouge_l)
            if "bert_ppl" in self.metrics:
                bert_ppl_score = ppl_score(pred_gen)
                evals.append(bert_ppl_score)


            eval_list.append([pred_gen, true_gen] + evals)

        eval_df = pd.DataFrame(eval_list,
                               columns=["pred",
                                        "true"] + self.metrics)
        return eval_df
