
import argparse
import os
import sys
from typing import List

import dill
import numpy as np
import pandas as pd
import sentencepiece as spm
from sumeval.metrics.rouge import RougeCalculator
from sumeval.metrics.bleu import BLEUCalculator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchtext.vocab import Vocab
from torchtext.data import Field, RawField, TabularDataset, BucketIterator

# from beam_search import Search, BeamSearch
import time

from models import LabelSmoothingLoss, FusionTransformerModel, SumEvaluator, denumericalize,type_label
from utils import Config
from bucket_iterator import BucketIterator
from data_utils import ABSADatesetReader
from nltk.translate.bleu_score import sentence_bleu


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare_default', default='./config/prepare_default.json', type=str)
    parser.add_argument('--train_default', default='./config/train_default.json', type=str)
    parser.add_argument('--generate_default', default='./config/generate_beam-5-06-3-60.json', type=str)

    parser.add_argument('--opinion_encoder_layers', default=1, type=int)
    parser.add_argument('--opinion_decoder_layers', default=3, type=int)
    parser.add_argument('--review_encoder_layers', default=4, type=int)
    parser.add_argument('--template_encoder_layers', default=4, type=int)
    parser.add_argument('--fusion_decoder_layers', default=6, type=int)
    parser.add_argument('--n_heads', default=2, type=int)
    parser.add_argument('--opinion_dim_feedforward', default=1024, type=int)
    parser.add_argument('--review_dim_feedforward', default=1024, type=int)
    parser.add_argument('--weight_OPR', default=1, type=float)
    parser.add_argument('--KL_weight', default=1, type=float)
    parser.add_argument('--use_keywords', default="yes", type=str)
    parser.add_argument('--K_V_dim', default=512, type=int)
    parser.add_argument('--opinion_max_len', default=6, type=int)
    parser.add_argument('--review_max_len', default=50, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--test_number', default=10, type=int)
    parser.add_argument('-- review_max_len', default=60, type=int)

    parser.add_argument('--model_name', default="./model/default_op2text_default_epoch-4.pt", type=str)
    # parser.add_argument('--model_name', default="./no_attention/default_op2text_default_epoch-4.pt", type=str)

    opt = parser.parse_args()

    p_conf = Config(opt.prepare_default)
    t_conf = Config(opt.train_default)
    g_conf = Config(opt.generate_default)
    assert p_conf.conf_type == "prepare"
    assert t_conf.conf_type == "train"
    # assert a_conf.conf_type == "aggregate"
    assert g_conf.conf_type == "generate"

    verbose = 0

    # Check if the method is valid
    assert g_conf["method"] in ["greedy", "beam"]

    # Basepath
    if "BASEPATH" not in os.environ:
        basepath = "."
    else:
        basepath = os.environ["BASEPATH"]

    # model filepath / output filepath
    model_filepath = opt.model_name
    # model_filepath = os.path.join(basepath,opt.model_name)
    print(model_filepath)
    output_filepath = os.path.join(basepath,
                                   "output",
                                   "{}_{}_op2text_{}_{}.csv".format(opt.model_name,p_conf.conf_name,
                                                                    t_conf.conf_name,
                                                                    g_conf.conf_name,))
    output_dirpath = os.path.dirname(output_filepath)

    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)

    # Load dataset
    # ================================================================
    absa_dataset = ABSADatesetReader(embed_dim=t_conf["text_embed_dim"],
                                     user_embed_dim=t_conf["user_embed_dim"], bus_embed_dim=t_conf["bus_embed_dim"],
                                     pos_embed_dim = t_conf["pos_embed_dim"],
                                     label_embed_dim =t_conf["label_embed_dim"],
                                     aspect_type_dim=t_conf["aspect_type_dim"],
                                     data_sets = "yelp",basepath = basepath,
                                    )
    out2pos = absa_dataset.out2pos_index
    padding_idx = absa_dataset.padding_idx
    bos_idx = absa_dataset.bos_idx
    eos_idx = absa_dataset.eos_idx
    unk_idx = absa_dataset.unk_idx
    pass_voc_size = absa_dataset.pass_list_embedding.shape[0]
    tgt_voc_size = absa_dataset.out_text_embedding.shape[0]
    print("Total Train number:{}".format(len(absa_dataset.train_data)))
    print("Total Dev number:{}".format(len(absa_dataset.dev_data)))
    print("Total Test number:{}".format(len(absa_dataset.test_data)))

    test_iterator = BucketIterator(absa_dataset.test_data,
                                   batch_size=opt.batch_size,
                                   sort=False, )
    #
    # trian_iterator = BucketIterator(absa_dataset.train_data,
    #                                batch_size=opt.batch_size,
    #                                sort=False, )
    #
    # dev_iterator = BucketIterator(absa_dataset.dev_data,
    #                                batch_size=opt.batch_size,
    #                                sort=False, )

    # We can use a different batch size for generation
    batch_size = opt.batch_size


    # ================================================================

    # Load model
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0,1,2,3" if USE_CUDA else "cpu")
    device = torch.device("cpu")
    model = FusionTransformerModel(absa_dataset.input_text_embedding,
                                   absa_dataset.pass_list_embedding,
                                   absa_dataset.out_text_embedding,
                                   absa_dataset.user_embedding,
                                   absa_dataset.business_embedding,
                                   absa_dataset.label_embedding,
                                   absa_dataset.aspect_type_embedding,
                                   absa_dataset.pos_embedding,
                                   n_heads=opt.n_heads,
                                   opinion_encoder_layers=opt.opinion_encoder_layers,
                                   opinion_decoder_layers=opt.opinion_decoder_layers,
                                   review_encoder_layers=opt.review_encoder_layers,
                                   template_encoder_layers=opt.template_encoder_layers,
                                   fusion_decoder_layers=opt.fusion_decoder_layers,
                                   dim_feedforward=opt.review_dim_feedforward,
                                   dropout=t_conf["model"]["params"]["dropout"],
                                   padding_idx=padding_idx,
                                   K_V_dim=opt.K_V_dim,
                                   device=device,
                                   opt=opt
                                   ).to(device)

    model.load_state_dict(torch.load(model_filepath,
                                     map_location=device))
    model.eval()

    # sumeval evaluator
    evaluator = SumEvaluator(metrics=t_conf["metrics"],
                             stopwords=False,
                             lang="en")

    # Old script used t_conf["training"]["gen_maxlen"]
    gen_maxlen = g_conf["gen_maxtoken"]


    ## 2. Generation for each review in test.csv

    print("Process individual revidws")
    opinion_max_len = opt.opinion_max_len
    review_max_len = opt.review_max_len
    eval_df_list = []
    beam_eval_df_list = []
    start_time = time.time()
    for batch_idx, batch in enumerate(test_iterator):

        print(batch_idx)
        start_time = time.time()
        # TODO: Switch generation ==================================
        if g_conf["method"] == "greedy":
            opinion_pred_list,review_pred,review_input = model.generate_all(
                batch["input_text"].to(device),
                batch["aspect_type"].to(device),
                batch["aspect_label"].to(device),
                batch["user"].to(device),
                batch["business"].to(device),
                batch["temp_pos"].to(device),
                out2pos,
                opinion_max_len,
                review_max_len,
                bos_index=bos_idx,
                pad_index=padding_idx,
                opt=opt,
            )
        elif g_conf["method"] == "beam":
            if "beam_width" in g_conf["params"]:
                beam_width = g_conf["params"]["beam_width"]
            else:
                beam_width = 3

            if "no_repeat_ngram_size" in g_conf["params"]:
                no_repeat_ngram_size = g_conf["params"]["no_repeat_ngram_size"]
            else:
                no_repeat_ngram_size = 0
            beam_width = 5
            opinion_tgt, review_tgt, review_input = model.generate_all_beamsearch(
                batch["input_text"].to(device),
                batch["aspect_type"].to(device),
                batch["aspect_label"].to(device),
                batch["user"].to(device),
                batch["business"].to(device),
                batch["temp_pos"].to(device),
                out2pos,
                opinion_max_len,
                review_max_len,
                bos_index=bos_idx,
                pad_index=padding_idx,
                opt=opt,
                vocab_size=tgt_voc_size,
                beam_size=beam_width,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        else:
            raise ValueError("Invalid decoding method: {}".format(g_conf["method"]))

        Test_review_label = batch["output_text"].to(device)
        true_gens = denumericalize(Test_review_label,
                                   absa_dataset.tokenizer_out.idx2word,
                                   join=" ")
        pred_gens = denumericalize(review_tgt,
                                   absa_dataset.tokenizer_out.idx2word,
                                   join=" ")


        # Generation evaluation
        eval_df = evaluator.eval(true_gens, pred_gens)
        eval_df_list.append(eval_df)

        if verbose == 1 and batch_idx % 100 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            start_time = end_time
            print("{} done ({:.2f} sec.)".format((batch_idx + 1) * batch_size,
                                                 elapsed_time))

    # Save results

        if batch_idx ==opt.test_number-1:
            print("---------------")
            break
    all_eval_df = pd.concat(eval_df_list, axis=0).reset_index(drop=True)
    all_eval_df.to_csv(output_filepath,
                       index=False)
