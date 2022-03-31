
import argparse
import os
import sys
from typing import List

import dill
import numpy as np
import pandas as pd

import torch
import time

from utils import Config
from bucket_iterator import BucketIterator
from data_utils import ABSADatesetReader
from models import LabelSmoothingLoss, SumEvaluator, denumericalize,type_label
from other_model import ExpansionNet
from torch.nn.utils.rnn import pad_sequence
hidden_size = 512
n_layers =1
attr_size = 64
attr_num = 2


if __name__ == "__main__":

    print("111111111")
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare_default', default='./config/prepare_default.json', type=str)
    parser.add_argument('--train_default', default='./config/train_default.json', type=str)
    parser.add_argument('--generate_default', default='./config/generate_beam-5-06-3-60.json', type=str)
    parser.add_argument('--opinion_encoder_layers', default=1, type=int)
    parser.add_argument('--opinion_decoder_layers', default=3, type=int)
    parser.add_argument('--review_encoder_layers', default=6, type=int)
    parser.add_argument('--template_encoder_layers', default=4, type=int)
    parser.add_argument('--fusion_decoder_layers', default=6, type=int)
    parser.add_argument('--n_heads', default=4, type=int)
    parser.add_argument('--opinion_dim_feedforward', default=1024, type=int)
    parser.add_argument('--review_dim_feedforward', default=1024, type=int)
    parser.add_argument('--weight_OPR', default=1, type=float)
    parser.add_argument('--KL_weight', default=1, type=float)
    parser.add_argument('--use_keywords', default="yes", type=str)
    parser.add_argument('--K_V_dim', default=512, type=int)
    parser.add_argument('--opinion_max_len', default=6, type=int)
    parser.add_argument('--review_max_len', default=50, type=int)
    parser.add_argument('--l2reg', default=0.0001, type=float)
    parser.add_argument('--learning_rate', default=0.00001, type=float)
    parser.add_argument('--optimizer', default='rmsprop', type=str)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--loss_func', default="cross_entropy", type=str)  # cross_entropy or label_smoothing

    parser.add_argument('--model_name', default="ENet", type=str)  # base_transformer
    parser.add_argument('--encoder_1_pt', default="default_op2text_defaultENet_encoder1_epoch-5.pt", type=str)
    parser.add_argument('--encoder_2_pt', default="default_op2text_defaultENet_encoder2_epoch-5.pt", type=str)
    parser.add_argument('--encoder_3_pt', default="default_op2text_defaultENet_encoder3_epoch-5.pt", type=str)
    parser.add_argument('--decoder_pt', default="default_op2text_defaultENet_decoder_epoch-5.pt", type=str)
    parser.add_argument('--test_number', default=10, type=int)

    parser.add_argument('--seed', default=3, type=int)
    parser.add_argument('--model_type', default="no_hier", type=str)
    parser.add_argument('--num_epoch', default=6, type=int)

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
    model_filepath = "./"+opt.model_name+".model"
    # model_filepath = os.path.join(basepath,opt.model_name)
    output_filepath = os.path.join(basepath,
                                   "output",
                                   "{}_{}.csv".format(opt.model_name,opt.encoder_1_pt))
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
    aspect_num = absa_dataset.aspect_type_embedding.shape[0]
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

    # We can use a different batch size for generation
    batch_size = opt.batch_size


    # ================================================================

    # Load model
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0,1,2,3" if USE_CUDA else "cpu")

    user_aspect_embeddings = np.zeros((absa_dataset.user_embedding.shape[0], attr_size))

    business_aspect_embeddings = np.zeros((absa_dataset.business_embedding.shape[0], attr_size))

    encoder1 = ExpansionNet.AttributeEncoder(absa_dataset.user_embedding, absa_dataset.business_embedding, hidden_size,
                                             attr_size, n_layers).to(device)
    print("--------")
    encoder2 = ExpansionNet.AttributeEncoder(user_aspect_embeddings, business_aspect_embeddings, hidden_size,
                                             aspect_num, n_layers).to(device)
    encoder3 = ExpansionNet.EncoderRNN(absa_dataset.out_text_embedding.shape[0], hidden_size,
                                       absa_dataset.input_text_embedding, n_layers=1).to(device)
    decoder = ExpansionNet.LuongAttnDecoderRNN('dot', absa_dataset.out_text_embedding, hidden_size, attr_size,
                                               absa_dataset.out_text_embedding.shape[0], n_layers).to(device)


    encoder_1_weight = os.path.join(model_filepath,opt.encoder_1_pt).replace("\\","/")

    encoder_2_weight = os.path.join(model_filepath, opt.encoder_2_pt).replace("\\", "/")
    encoder_3_weight = os.path.join(model_filepath, opt.encoder_3_pt).replace("\\", "/")
    decoder_weight = os.path.join(model_filepath, opt.decoder_pt).replace("\\", "/")


    encoder1.load_state_dict(torch.load( encoder_1_weight,
                                     map_location=device))
    encoder2.load_state_dict(torch.load(encoder_2_weight,
                                        map_location=device))
    encoder3.load_state_dict(torch.load(encoder_3_weight,
                                        map_location=device))
    decoder.load_state_dict(torch.load(decoder_weight,
                                        map_location=device))
    encoder1.eval()
    # encoder2.eval()
    encoder3.eval()
    decoder.eval()

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

        key_list = []
        key = batch["input_text"].view(batch["input_text"].shape[0], -1)
        for i in range(key.shape[0]):
            key_pad = key[i, :] != absa_dataset.padding_idx
            key_list.append(key[i, :].mul(key_pad))
        key_input = pad_sequence(key_list, padding_value=0, batch_first=True)

        if "beam_width" in g_conf["params"]:
            beam_width = g_conf["params"]["beam_width"]
        else:
            beam_width = 3

        if "no_repeat_ngram_size" in g_conf["params"]:
            no_repeat_ngram_size = g_conf["params"]["no_repeat_ngram_size"]
        else:
            no_repeat_ngram_size = 0
        beam_width = 3

        review_tgt = ExpansionNet.generate_all_beamsearch(
            key.to(device),
            batch["user"].to(device),
            batch["business"].to(device),
            encoder1,
            encoder3,
            decoder,
            review_max_len,
            opt=opt,
            vocab_size = tgt_voc_size,
            beam_size=3,
            no_repeat_ngram_size=0,
            device=device
        )
        print(review_tgt.shape)
        exit()
        review_tgt= torch.argmax(review_tgt, dim=-1)

        Test_review_label = batch["output_text"].to(device)

        true_gens = denumericalize(Test_review_label,
                                   absa_dataset.tokenizer_out.idx2word,
                                   join=" ")
        print(review_tgt)
        pred_gens = denumericalize(review_tgt,
                                   absa_dataset.tokenizer_out.idx2word,
                                   join=" ")

        keyword = type_label(batch["input_text"].data,absa_dataset.tokenizer_input.idx2word)

        label = type_label(batch["aspect_label"].data, absa_dataset.tokenizer_aspect_label.idx2dependency)
        keyword_list = {"aspect_word":keyword}
        label_list = {"label":label}
        keyword_list = pd.DataFrame(keyword_list)
        label_list = pd.DataFrame(label)
        print(keyword_list)
        print(label_list)


        eval_df = evaluator.eval(true_gens, pred_gens)
        eval_df = pd.concat([eval_df, keyword_list, label_list], axis=1)
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
