import os
import sys

import pandas as pd
import sentencepiece as spm
import  numpy
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.optim import SGD
# from torchtext.data import Field, TabularDataset, BucketIterator
from models import LabelSmoothingLoss, SumEvaluator, denumericalize,type_label,FusionTransformerModel
from utils import Config
from bucket_iterator import BucketIterator
from data_utils import ABSADatesetReader

import  re
# import  tqdm
import argparse
import  time
import datetime
from tqdm import tqdm
# from apex import  amp
import codecs

time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

# from beam_search import Search, BeamSearch
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import  numpy as np
import  random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

















if __name__ == "__main__":
    print("111111111")
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare_default', default='./config/prepare_default.json', type=str)
    parser.add_argument('--train_default', default='./config/train_default.json', type=str)
    parser.add_argument('--opinion_encoder_layers', default=1,type=int)
    parser.add_argument('--opinion_decoder_layers', default=3, type=int)
    parser.add_argument('--review_encoder_layers', default=6, type=int)
    parser.add_argument('--template_encoder_layers', default=4, type=int)
    parser.add_argument('--fusion_decoder_layers', default=6, type=int)
    parser.add_argument('--n_heads', default=4, type=int)
    parser.add_argument('--opinion_dim_feedforward', default=1024, type=int)
    parser.add_argument('--review_dim_feedforward', default=1024, type=int)
    parser.add_argument('--weight_OPR', default=5, type=float)
    parser.add_argument('--KL_weight', default=1, type=float)
    parser.add_argument('--use_keywords', default="yes", type=str)
    parser.add_argument('--K_V_dim', default=512, type=int)
    parser.add_argument('--opinion_max_len', default=6, type=int)
    parser.add_argument('--review_max_len', default=50, type=int)
    parser.add_argument('--l2reg', default=0.0001, type=float)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--loss_func', default="cross_entropy", type=str)
    parser.add_argument('--seed', default=5, type=int)
    parser.add_argument('--num_epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=3, type=int)
    # parser.add_argument('--model_name', default="Base_Transformer", type=str)

    parser.add_argument('--model_name', default="Hier_Transformer", type=str)



    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }


    opt = parser.parse_args()
    # p_conf = Config(sys.argv[1])
    # t_conf = Config(sys.argv[2])
    p_conf = Config(opt.prepare_default)
    t_conf = Config(opt.train_default)
    setup_seed(opt.seed)

    assert p_conf.conf_type == "prepare"
    assert t_conf.conf_type == "train"
    verbose = 4

    data_split_ratio = [0.8, 0.1, 0.1]
    # pretrained_vectors = None
    hyper_setting_name = ""
    for key,value  in opt.__dict__.items():
        hyper_setting_name+="{0}: {1} ".format(key,value)
    print(hyper_setting_name)


    loss_func = opt.loss_func # cross_entropy_avgsum
    sp_model_filepath = None  # "model/hm_model.model"

    batch_size = opt.batch_size
    num_epoch = opt.num_epoch
    if "clipping" in t_conf["training"]:
        clipping = t_conf["training"]["clipping"]
    else:
        clipping = None

    gen_maxlen = opt.review_max_len
    metrics = t_conf["metrics"]
    if "BASEPATH" not in os.environ:
        basepath = ""
    else:
        basepath = os.environ["BASEPATH"]
    model_filepath = os.path.join(basepath,
                                  "{}".format(opt.model_name),
                                  "{}_op2text_{}.pt".format(p_conf.conf_name,
                                                            t_conf.conf_name))

    # Dirpath
    model_dirname = os.path.dirname(model_filepath)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)

    # Data file
    data_dirpath = os.path.join(basepath,
                                "./data/yelp-default/"
                                )
    data_dirpath = data_dirpath.replace("\\", "/")

    train_filepath = os.path.join(data_dirpath,
                                  "train.json").replace("\\", "/")
    valid_filepath = os.path.join(data_dirpath,
                                  "dev.json").replace("\\", "/")
    # print(train_filepath)

    assert os.path.exists(train_filepath)
    assert os.path.exists(valid_filepath)

    # config
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    absa_dataset = ABSADatesetReader(embed_dim=t_conf["text_embed_dim"],
                                     user_embed_dim=t_conf["user_embed_dim"], bus_embed_dim=t_conf["bus_embed_dim"],
                                     pos_embed_dim = t_conf["pos_embed_dim"],
                                     label_embed_dim =t_conf["label_embed_dim"],
                                     aspect_type_dim=t_conf["aspect_type_dim"],
                                     data_sets = "yelp",basepath = basepath,
                                    )
    padding_idx = absa_dataset.padding_idx
    bos_idx = absa_dataset.bos_idx
    eos_idx = absa_dataset.eos_idx
    unk_idx = absa_dataset.unk_idx
    out2pos = absa_dataset.out2pos_index
    pass_voc_size = absa_dataset.pass_list_embedding.shape[0]
    tgt_voc_size = absa_dataset.out_text_embedding.shape[0]
    print("Total Train number:{}".format(len(absa_dataset.train_data)))
    print("Total Dev number:{}".format(len(absa_dataset.dev_data)))
    print("Total Test number:{}".format(len(absa_dataset.test_data)))
    train_iterator = BucketIterator(absa_dataset.train_data,
                                    batch_size=batch_size,
                                    sort=True,
                                    )

    valid_iterator = BucketIterator(absa_dataset.dev_data,
                                    batch_size=batch_size,
                                    sort=False,
                                    )

    test_iterator = BucketIterator(absa_dataset.test_data,
                                    batch_size=batch_size,
                                    sort=False,)

    evaluator = SumEvaluator(metrics=metrics,
                             stopwords=False,
                             lang="en")


    # Transformer model

    label_weight = {absa_dataset.pos_label_idx:0.5,absa_dataset.neu_label_idx:1,
                    absa_dataset.neg_label_idx:2,absa_dataset.padding_idx:0,}


    model = FusionTransformerModel(absa_dataset.input_text_embedding,
                                   absa_dataset.pass_list_embedding,
                                   absa_dataset.out_text_embedding,
                                   absa_dataset.user_embedding,
                                   absa_dataset.business_embedding,
                                   absa_dataset.label_embedding,
                                   absa_dataset.aspect_type_embedding,
                                   absa_dataset.pos_embedding,
                                   n_heads = opt.n_heads,
                                   opinion_encoder_layers = opt.opinion_encoder_layers,
                                   opinion_decoder_layers = opt.opinion_decoder_layers,
                                   review_encoder_layers=opt.review_encoder_layers,
                                   template_encoder_layers = opt.template_encoder_layers,
                                   fusion_decoder_layers = opt.fusion_decoder_layers,
                                   dim_feedforward=opt.review_dim_feedforward,
                                   dropout=t_conf["model"]["params"]["dropout"],
                                   padding_idx=padding_idx,
                                   K_V_dim=opt.K_V_dim,
                                   device =device,
                                   opt = opt
                                   ).to(device)

    # model = nn.DataParallel(FusionTransformerModel(absa_dataset.input_text_embedding,
    #                                absa_dataset.pass_list_embedding,
    #                                absa_dataset.out_text_embedding,
    #                                absa_dataset.user_embedding,
    #                                absa_dataset.business_embedding,
    #                                absa_dataset.label_embedding,
    #                                absa_dataset.aspect_type_embedding,
    #                                absa_dataset.pos_embedding,
    #                                n_heads = opt.n_heads,
    #                                opinion_encoder_layers = opt.opinion_encoder_layers,
    #                                opinion_decoder_layers = opt.opinion_decoder_layers,
    #                                review_encoder_layers=opt.review_encoder_layers,
    #                                template_encoder_layers = opt.template_encoder_layers,
    #                                fusion_decoder_layers = opt.fusion_decoder_layers,
    #                                dim_feedforward=opt.review_dim_feedforward,
    #                                dropout=t_conf["model"]["params"]["dropout"],
    #                                padding_idx=padding_idx,
    #                                K_V_dim=opt.K_V_dim,
    #                                device =device,
    #                                opt = opt)).to(device)

    # Optimizer
    # General template to make an optimzier instance
    # e.g.,)

    # optimizer = eval("{}(model.parameters(), **{})".format(t_conf["training"]["optimizer"]["cls"],
    #                                                        str(t_conf["training"]["optimizer"]["params"])))
    _params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optimizers[opt.optimizer](_params,lr = opt.learning_rate,weight_decay=opt.l2reg)
    # optimizer = optimizers[opt.optimizer](_params,
    #                                       lr=opt.learning_rate,
    #                                       momentum=0.9,
    #                                       weight_decay =opt.l2reg,
    #                                       nesterov=True)

    scheduler = StepLR(optimizer,
                       step_size=2,
                       gamma=0.1)
    # model,optimizer = amp.initialize(model,optimizer,opt_level="O1")

    if loss_func == "cross_entropy":

        opinion_criterion= nn.CrossEntropyLoss(ignore_index=padding_idx,reduction="none").to(device)  # mean
        review_criterion = nn.CrossEntropyLoss(ignore_index=padding_idx,
                                        reduction="none").to(device)

    elif loss_func == "label_smoothing":
        opinion_criterion = LabelSmoothingLoss(label_smoothing=0.1,
                                       tgt_vocab_size=pass_voc_size,
                                       device=device,
                                       ignore_index=padding_idx)

        review_criterion = LabelSmoothingLoss(label_smoothing=0.1,
                                               tgt_vocab_size=tgt_voc_size,
                                               device=device,
                                               ignore_index=padding_idx)
        # print(pass_voc_size)
        # print(tgt_voc_size)

    log_data_list = []
    # Training
    for epoch in range(num_epoch):
        time_start = time.time()
        OP_Loss = 0.0
        RE_Loss = 0.0
        weight_OPR = opt.weight_OPR
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        scheduler.step()
        print("Epoch:", epoch, "LR:", scheduler.get_lr())
        # train_iterator = tqdm(train_iterator)
        for batch_idx, batch in enumerate(tqdm(train_iterator)):
            # if verbose == 1 and batch_idx % 100 == 0:
            #     print(batch_idx)

            # exit()
            b_size,type_num,passage_seq_len = batch["passage_list"].shape
            # print(batch["passage_list"].shape)
            # # exit()
            # print("-"*30)
            b_size, tgt_seq_len = batch["output_text"].shape
            optimizer.zero_grad()

            opinion_outputs_list, review_outputs,review_input,KL_loss_reconstruct = model(
                batch["input_text"].to(device),
                batch["aspect_type"].to(device),
                         batch["aspect_label"].to(device),
                         batch["user"].to(device),
                         batch["business"].to(device),
                         batch["passage_list"].to(device),
                         batch["temp_pos"].to(device),
                         batch["output_text"].to(device),
                        batch["output_pos"].to(device),

                         )
            review_pred  = review_outputs
            review_label = batch["output_text"].to(device)
            review_label = torch.cat([review_label[:, 1:],
                                    torch.LongTensor(b_size,1).fill_(0).to(device)], axis=-1)


            OP_loss_vals = []
            if loss_func == "cross_entropy":
                for i in range(type_num):
                    passage_label_i = batch["passage_list"][:,i,:].to(device)
                    passage_label_i = torch.cat([passage_label_i[:, 1:],
                                    torch.LongTensor(b_size,1).fill_(0).to(device)], axis=1)

                    passage_pred_i = opinion_outputs_list[i].to(device)

                    # 增加neg权重

                    passage_i_loss_list = review_criterion(passage_pred_i.transpose(1, 2),
                                             passage_label_i)
                    # weight_tensor = torch.FloatTensor([label_weight[i] for i in batch["aspect_label"][:,i].numpy()]).unsqueeze(-1)
                    # weight_tensor = weight_tensor.repeat(1,passage_i_loss_list.shape[1]).to(device)
                    passage_i_loss=passage_i_loss_list.sum(axis=0).mean()
                    # passage_i_loss = torch.mul(passage_i_loss_list,weight_tensor).sum(axis=0).mean()
                    OP_loss_vals.append(passage_i_loss)
                OP_Loss = sum(OP_loss_vals)
                #Rview_loss
                RE_loss_vals = review_criterion(review_pred.transpose(1, 2),
                                      review_label)
                RE_Loss = RE_loss_vals.sum(axis=0).mean()
            elif loss_func == "label_smoothing":
                for i in range(type_num):
                    passage_label_i = batch["passage_list"][:,i,:].to(device)
                    passage_label_i = torch.cat([passage_label_i[:, 1:],
                                    torch.LongTensor(b_size,1).fill_(0).to(device)], axis=1)
                    passage_pred_i = opinion_outputs_list[i].to(device)

                    # print(passage_pred_i.shape)
                    # print(passage_label_i.shape)

                    passage_loss_list = opinion_criterion(passage_pred_i, passage_label_i)


                    # weight_tensor = torch.FloatTensor(
                        # [label_weight[i] for i in batch["aspect_label"][:, i].numpy()]).unsqueeze(-1)
                    #
                    # weight_tensor = weight_tensor.repeat(1, passage_loss_list.shape[1]).to(device)

                    # passage_i_loss = torch.mul(passage_loss_list, weight_tensor).sum(1).mean()
                    passage_i_loss = passage_loss_list.sum(axis=0).mean()
                    OP_loss_vals.append(passage_i_loss)
                OP_Loss = sum(OP_loss_vals)

                RE_Loss= review_criterion(review_pred,review_label).sum(1).mean()

            loss = RE_Loss+weight_OPR*OP_Loss+opt.KL_weight*KL_loss_reconstruct

            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            loss.backward()
            # break
            if clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               clipping)
            optimizer.step()
            loss_val = loss.data.item()  # * batch.in_text.size(0)

            # exit()
            if verbose >= 2 and batch_idx % 1000 == 0:
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                print("Train: {} LR:{} Total_loss={}; OP_Loss={}; RE_loss={};KL_loss={}".format(batch_idx,lr,loss_val,OP_Loss*weight_OPR,RE_Loss,KL_loss_reconstruct))

                # print(review_input)
                print("True: {}".format(denumericalize(review_label,absa_dataset.tokenizer_out.idx2word)[0]))
                print("Pred: {}".format(denumericalize(review_pred.argmax(dim=-1),
                                                       absa_dataset.tokenizer_out.idx2word)[0]))

                print("aspect_type:{}   Label:{}".format(type_label(batch["aspect_type"].data,
                     absa_dataset.tokenizer_type.idx2word)[0],type_label(batch["aspect_label"].data,absa_dataset.tokenizer_aspect_label.idx2dependency)[0]))

                print("passage_true:{} ".format(type_label(batch["passage_list"][0,:, :].data,absa_dataset.tokenizer_passage.idx2word)))


                batch_passage_pred =torch.cat([ i[0,:,:].argmax(dim=-1).unsqueeze(0) for i in opinion_outputs_list],dim=0)

                print("passage_pred:{}".format(type_label(batch_passage_pred.data,absa_dataset.tokenizer_passage.idx2word)))
                print("review_input:{}".format(type_label(review_input.data,absa_dataset.tokenizer_passage.idx2word)))


            training_loss += loss_val

        training_loss /= batch_idx+1

        # Validation
        model.eval()
        time_end = time.time()
        time_consule = time_end - time_start
        print("----------this epoch time :{0}".format(str(datetime.timedelta(seconds=time_consule))))

        eval_df_list = []
        beam_eval_df_list = []
        greedy_time = 0.
        beamsearch_time = 0.


        valid_loss = 0.00
        for batch_idx, batch in enumerate(tqdm(valid_iterator)):

            b_size, val_type_num, val_passage_seq_len = batch["passage_list"].shape
            Val_opinion_outputs_list, Val_review_outputs,Val_review_input,Val_KL_loss_reconstruct= model(
                batch["input_text"].to(device),
                batch["aspect_type"].to(device),
                     batch["aspect_label"].to(device),
                     batch["user"].to(device),
                     batch["business"].to(device),
                     batch["passage_list"].to(device),
                     batch["temp_pos"].to(device),
                     batch["output_text"].to(device),
                    batch["output_pos"].to(device),
                     )
            Val_review_pred = Val_review_outputs
            Val_review_label = batch["output_text"].to(device)
            Val_review_label = torch.cat([Val_review_label[:, 1:],
                                      torch.LongTensor(b_size, 1).fill_(0).to(device)], axis=-1)
            Val_OP_loss_vals = []
            if loss_func == "cross_entropy":
                for i in range(val_type_num):
                    # print(batch["passage_list"].shape)
                    # print(Val_review_input)
                    Val_passage_label_i = batch["passage_list"][:, i, :].to(device)
                    Val_passage_label_i = torch.cat([Val_passage_label_i[:, 1:],
                                                 torch.LongTensor(b_size, 1).fill_(0).to(device)], axis=1)

                    Val_passage_pred_i = Val_opinion_outputs_list[i].to(device)
                    Val_passage_loss_list = opinion_criterion(Val_passage_pred_i.transpose(1, 2),
                              Val_passage_label_i)
                    #
                    # weight_tensor = torch.FloatTensor(
                    #     [label_weight[i] for i in batch["aspect_label"][:, i].numpy()]).unsqueeze(-1)

                    # weight_tensor = weight_tensor.repeat(1, Val_passage_loss_list.shape[1]).to(device)
                    # Val_passage_loss = Val_passage_loss_list.sum(axis=0).mean()
                    # Val_passage_loss = torch.mul(Val_passage_loss_list,weight_tensor).sum(axis=0).mean()
                    Val_passage_loss = Val_passage_loss_list.sum(axis=0).mean()
                    Val_OP_loss_vals.append(Val_passage_loss)
                    # Avg of sum of token loss (after ignoring padding tokens)
                    # loss = loss_vals
                Val_OP_Loss =sum(Val_OP_loss_vals)

                Val_RE_loss_vals = review_criterion(Val_review_pred.transpose(1, 2),
                                         Val_review_label)
                # Avg of sum of token loss (after ignoring padding tokens)
                # loss = loss_vals
                Val_RE_Loss = Val_RE_loss_vals.sum(axis=0).mean()


            elif loss_func == "label_smoothing":
                for i in range(val_type_num):
                    Val_passage_label_i = batch["passage_list"][:, i, :].to(device)
                    Val_passage_label_i = torch.cat([Val_passage_label_i[:, 1:],
                                                 torch.LongTensor(b_size, 1).fill_(0).to(device)], axis=1)
                    Val_passage_pred_i= Val_opinion_outputs_list[i].to(device)

                    Val_passage_loss_list = opinion_criterion(Val_passage_pred_i, Val_passage_label_i)

                    # weight_tensor = torch.FloatTensor(
                    #     [label_weight[i] for i in batch["aspect_label"][:, i].numpy()]).unsqueeze(-1)
                    #
                    # weight_tensor = weight_tensor.repeat(1, Val_passage_loss_list.shape[1]).to(device)
                    # Val_passage_loss = torch.mul(Val_passage_loss_list, weight_tensor).sum(-1).mean()
                    Val_passage_loss = Val_passage_loss_list.sum(axis=0).mean()

                    Val_OP_loss_vals.append(Val_passage_loss)
                Val_OP_Loss = sum(Val_OP_loss_vals)
                Val_RE_Loss = review_criterion(Val_review_pred, Val_review_label).sum(1).mean()

            Val_loss = Val_RE_Loss + weight_OPR * Val_OP_Loss+opt.KL_weight*Val_KL_loss_reconstruct

            valid_loss_val = Val_loss.data.item() * batch["aspect_label"].shape[0]
            valid_loss += valid_loss_val

            # if batch_idx % 10000 == 0:
            #     opinion_max_len = opt.opinion_max_len
            #     review_max_len = opt.review_max_len
            #     # opinion_pred_list,review_pred,review_input = model.generate_all(
            #     #     batch["input_text"].to(device),
            #     #     batch["aspect_type"].to(device),
            #     #     batch["aspect_label"].to(device),
            #     #     batch["user"].to(device),
            #     #     batch["business"].to(device),
            #     #     batch["temp_pos"].to(device),
            #     #     opinion_max_len,
            #     #     review_max_len,
            #     #     bos_index=bos_idx,
            #     #     pad_index=padding_idx,
            #     #     opt=opt,
            #     # )
            #
            #     opinion_pred_list,review_pred,review_input = model.generate_all_beamsearch(
            #         batch["input_text"].to(device),
            #         batch["aspect_type"].to(device),
            #         batch["aspect_label"].to(device),
            #         batch["user"].to(device),
            #         batch["business"].to(device),
            #         batch["temp_pos"].to(device),
            #         out2pos,
            #         opinion_max_len,
            #         review_max_len,
            #         bos_index=bos_idx,
            #         pad_index=padding_idx,
            #         opt=opt,
            #         vocab_size = tgt_voc_size,
            #         beam_size=3,
            #         no_repeat_ngram_size=0,
            #     )
            #
            #     print("----------------------------Val_Text-----------------------------------")
            #     # print(review_input)
            #     print("True: {}".format(denumericalize(Val_review_label, absa_dataset.tokenizer_out.idx2word)[0]))
            #     print("Pred: {}".format(denumericalize(review_pred,
            #                                            absa_dataset.tokenizer_out.idx2word)[0]))
            #
            #     print("aspect_type:{}   Label:{}".format(type_label(batch["aspect_type"].data,
            #                                                         absa_dataset.tokenizer_type.idx2word)[0],
            #                                              type_label(batch["aspect_label"].data,
            #                                                         absa_dataset.tokenizer_aspect_label.idx2dependency)[
            #                                                  0]))
            #
            #
            #     print("passage_true:{} ".format(type_label(batch["passage_list"][0,:, :].data,absa_dataset.tokenizer_passage.idx2word)))
            #     # print(opinion_pred_list[0].shape)
            #     Val_batch_passage_pred = torch.cat([ i[0,:].unsqueeze(0) for i in opinion_pred_list],dim=0)
            #
            #     print("passage_pred:{}".format(type_label(Val_batch_passage_pred.data,
            #                    absa_dataset.tokenizer_passage.idx2word)))
            #     print("review_input:{}".format(type_label(review_input.data, absa_dataset.tokenizer_passage.idx2word)))

        valid_loss /= batch_idx+1
        print("\n")

        print('---------------------------------------------Epoch: {}, Training loss: {:.2f}, Valid loss: {:.2f}---------------------------'.format(
            epoch, training_loss, valid_loss))
        log_data_list.append([epoch,
                              training_loss,
                              valid_loss])
        if epoch==0:
            f_out = codecs.open('log/' + opt.model_name + '_' + '_val.txt', 'a+', encoding="utf-8")
            f_out.write('time:{0}\n'.format(time_str))
            f_out.write(hyper_setting_name)

        # TODO(Yoshi): The last model is redundant
        torch.save(model.state_dict(),
                   model_filepath.replace(".pt", "_epoch-{}.pt".format(epoch)))

    torch.save(model.state_dict(),
               model_filepath)

    # with open(model_filepath.replace(".pt", "_IN_TEXT.field"), "wb") as fout:
    #     dill.dump(IN_TEXT, fout)
    # with open(model_filepath.replace(".pt", "_OUT_TEXT.field"), "wb") as fout:
    #     dill.dump(OUT_TEXT, fout)
    # with open(model_filepath.replace(".pt", "_ID.field"), "wb") as fout:
    #     dill.dump(ID, fout)

    # Write out log
    df = pd.DataFrame(log_data_list,
                      columns=["epoch", "training_loss", "valid_loss"])
    df.to_csv(model_filepath.replace(".pt", "_loss.csv"))
