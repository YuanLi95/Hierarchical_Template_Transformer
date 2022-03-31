
# raise ValueError("deal with Variable requires_grad, and .cuda()")
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.backends.cudnn as cudnn
import argparse
from beam_search import BeamSearch, NgramBlocking
import  numpy as np
import itertools
import random
import math
import sys
import os
from tqdm import tqdm
from models import LabelSmoothingLoss, SumEvaluator, denumericalize,type_label
from utils import Config
from other_model import ExpansionNet
from bucket_iterator import BucketIterator
from data_utils import ABSADatesetReader
import  time
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils.rnn import pad_sequence
import  datetime
import codecs
import pandas as pd
hidden_size = 512
n_layers =1
attr_size = 64
attr_num = 2
cudnn.benchmark = True




# from beam_search import Search, BeamSearch
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

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
    parser.add_argument('--weight_OPR', default=1, type=float)
    parser.add_argument('--KL_weight', default=1, type=float)
    parser.add_argument('--use_keywords', default="yes", type=str)
    parser.add_argument('--K_V_dim', default=512, type=int)
    parser.add_argument('--opinion_max_len', default=6, type=int)
    parser.add_argument('--review_max_len', default=50, type=int)
    parser.add_argument('--l2reg', default=0.0001, type=float)
    parser.add_argument('--learning_rate', default=0.00001, type=float)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--batch_size', default=3, type=int)
    parser.add_argument('--loss_func', default="cross_entropy", type=str) #cross_entropy or label_smoothing
    parser.add_argument('--model_name', default="ENet", type=str)  #base_transformer
    parser.add_argument('--seed', default=3, type=int)
    parser.add_argument('--model_type', default="hier", type=str)
    parser.add_argument('--num_epoch', default=6, type=int)






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
    setup_seed(opt.seed)

    hyper_setting_name = ""
    for key,value  in opt.__dict__.items():
        hyper_setting_name+="{0}: {1} ".format(key,value)
    print(hyper_setting_name)

    # p_conf = Config(sys.argv[1])
    # t_conf = Config(sys.argv[2])
    p_conf = Config(opt.prepare_default)
    t_conf = Config(opt.train_default)

    assert p_conf.conf_type == "prepare"
    assert t_conf.conf_type == "train"
    verbose = 4

    data_split_ratio = [0.8, 0.1, 0.1]
    # pretrained_vectors = None



    # loss_func = t_conf["training"]["loss_func"]  # cross_entropy_avgsum
    sp_model_filepath = None  # "model/hm_model.model"
    loss_func = opt.loss_func

    # batch_size = t_conf["training"]["batch_size"]
    num_epoch = t_conf["training"]["num_epoch"]
    if "clipping" in t_conf["training"]:
        clipping = t_conf["training"]["clipping"]
    else:
        clipping = None

    gen_maxlen = t_conf["training"]["gen_maxlen"]
    metrics = t_conf["metrics"]
    if "BASEPATH" not in os.environ:
        basepath = ""
    else:
        basepath = os.environ["BASEPATH"]
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    model_filepath = os.path.join(basepath,
                                  "{}.model".format(opt.model_name),
                                  "{}_op2text_{}.pt".format(p_conf.conf_name,
                                                            t_conf.conf_name))
    model_filepath = model_filepath.replace("\\", "/")
    model_filepath = model_filepath.replace(":", "")
    model_filepath = model_filepath.replace("-", "_")
    model_filepath = model_filepath.replace(" ", "")

    print(model_filepath)

    # Dirpath
    model_dirname = os.path.dirname(model_filepath)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)

    # Data file
    data_dirpath = os.path.join(basepath,
                                "./data/yelp-default/example/"
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
    device = torch.device("cuda:0,1,2,3" if USE_CUDA else "cpu")
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
    # out2pos = absa_dataset.out2pos_index
    pass_voc_size = absa_dataset.pass_list_embedding.shape[0]
    tgt_voc_size = absa_dataset.out_text_embedding.shape[0]
    print("Total Train number:{}".format(len(absa_dataset.train_data)))
    print("Total Dev number:{}".format(len(absa_dataset.dev_data)))
    print("Total Test number:{}".format(len(absa_dataset.test_data)))
    train_iterator = BucketIterator(absa_dataset.train_data,
                                    batch_size=opt.batch_size,
                                    sort=True,
                                    )

    valid_iterator = BucketIterator(absa_dataset.dev_data,
                                    batch_size=opt.batch_size,
                                    sort=False,
                                    )

    test_iterator = BucketIterator(absa_dataset.test_data,
                                    batch_size=opt.batch_size,
                                    sort=False,)

    evaluator = SumEvaluator(metrics=metrics,
                             stopwords=False,
                             lang="en")


    # Transformer model

    aspect_num = absa_dataset.aspect_type_embedding.shape[0]
    # optimizer = eval("{}(model.parameters(), **{})".format(t_conf["training"]["optimizer"]["cls"],
    #                                                        str(t_conf["training"]["optimizer"]["params"])))

    user_aspect_embeddings =  np.zeros((absa_dataset.user_embedding.shape[0],attr_size))

    business_aspect_embeddings = np.zeros((absa_dataset.business_embedding.shape[0],attr_size ))

    encoder1 = ExpansionNet.AttributeEncoder(absa_dataset.user_embedding,absa_dataset.business_embedding,hidden_size,attr_size, n_layers).to(device)
    print("--------")
    encoder2 = ExpansionNet.AttributeEncoder(user_aspect_embeddings,business_aspect_embeddings,hidden_size,aspect_num, n_layers).to(device)
    encoder3 = ExpansionNet.EncoderRNN(absa_dataset.out_text_embedding.shape[0], hidden_size, absa_dataset.input_text_embedding, n_layers=1).to(device)
    decoder = ExpansionNet.LuongAttnDecoderRNN('dot', absa_dataset.out_text_embedding, hidden_size,attr_size, absa_dataset.out_text_embedding.shape[0], n_layers).to(device)

    encoder1_optimizer = optim.Adam(encoder1.parameters(), lr = opt.learning_rate)
    encoder2_optimizer = optim.Adam(encoder2.parameters(), lr=opt.learning_rate)
    encoder3_optimizer = optim.Adam(encoder3.parameters(), lr=opt.learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=opt.learning_rate)

    scheduler_1 = StepLR(encoder1_optimizer,
                       step_size=2,
                       gamma=0.1)
    scheduler_2 = StepLR(encoder2_optimizer,
                       step_size=2,
                       gamma=0.1)
    scheduler_3 = StepLR(encoder3_optimizer,
                       step_size=2,
                       gamma=0.1)
    scheduler_4 = StepLR(decoder_optimizer,
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
                                       ignore_index=padding_idx,
                                       reduction="none",
                                               )

        review_criterion = LabelSmoothingLoss(label_smoothing=0.1,
                                               tgt_vocab_size=tgt_voc_size,
                                               device=device,
                                               ignore_index=padding_idx,
                                               reduction = "none",
                                              )
        # print(pass_voc_size)
        # print(tgt_voc_size)

    log_data_list = []
    # Training
    for epoch in range(opt.num_epoch):
        time_start = time.time()
        OP_Loss = 0.0
        RE_Loss = 0.0
        weight_OPR = opt.weight_OPR
        training_loss = 0.0
        valid_loss = 0.0
        encoder1.train()
        encoder2.train()
        encoder3.train()
        decoder.train()

        for batch_idx, batch in enumerate(tqdm(train_iterator)):

            b_size,type_num,passage_seq_len = batch["passage_list"].shape
            # print(batch["input_text"])
            key_list = []
            key = batch["input_text"].view(batch["input_text"].shape[0],-1)
            for i in range(key.shape[0]):
                key_pad =  key[i,:] != absa_dataset.padding_idx
                key_list.append(key[i,:].mul(key_pad))
            key_input = pad_sequence(key_list, padding_value=0, batch_first=True)

            review_outputs = ExpansionNet.run(
                        key_input.to(device),
                         batch["user"].to(device),
                         batch["business"].to(device),
                        batch["output_text"].to(device),
                        encoder1,
                        encoder2,
                        encoder3,
                        decoder,
                        encoder1_optimizer,
                        encoder2_optimizer,
                        encoder3_optimizer,
                        decoder_optimizer,
                        device=device,
                        max_length=50
                         )
            review_pred  = torch.argmax(review_outputs,dim=-1)


            review_label = batch["output_text"].to(device)
            review_label = torch.cat([review_label[:, 1:],
                                    torch.LongTensor(b_size,1).fill_(0).to(device)], axis=-1)

            if loss_func == "cross_entropy":
                #Rview_loss
                RE_loss_vals = review_criterion(review_outputs.transpose(1, 2),
                                      review_label)
                RE_Loss = RE_loss_vals.sum(axis=0).mean()
            elif loss_func == "label_smoothing":

                # 增加neg权重


                RE_Loss= review_criterion(review_outputs,review_label).sum(-1).mean()

            loss = RE_Loss

            loss.backward()
            encoder1_optimizer.step()
            encoder3_optimizer.step()
            decoder_optimizer.step()


            loss_val = loss.data.item()  # * batch.in_text.size(0)

            # exit()
            if verbose >= 2 and batch_idx % 1000 == 0:
                lr = encoder1_optimizer.state_dict()['param_groups'][0]['lr']
                print("Train: {} LR:{} Total_loss={}".format(batch_idx,lr,loss_val))

                # print(review_input)
                print("True: {}".format(denumericalize(review_label,absa_dataset.tokenizer_out.idx2word)[0]))
                print("Pred: {}".format(denumericalize(review_pred,
                                                       absa_dataset.tokenizer_out.idx2word)[0]))

                print("aspect_type:{}   Label:{}".format(type_label(batch["aspect_type"].data,
                     absa_dataset.tokenizer_type.idx2word)[0],type_label(batch["aspect_label"].data,absa_dataset.tokenizer_aspect_label.idx2dependency)[0]))


            training_loss += loss_val

        training_loss /= batch_idx

        encoder1.eval()
        encoder2.eval()
        encoder3.eval()
        decoder.eval()

        time_end = time.time()
        time_consule = time_end - time_start
        print("----------this epoch time :{0}".format(str(datetime.timedelta(seconds=time_consule))))

        eval_df_list = []
        beam_eval_df_list = []
        greedy_time = 0.
        beamsearch_time = 0.


        valid_loss = 0.00
        for batch_idx, batch in enumerate(tqdm(valid_iterator)):
            key_list = []
            key = batch["input_text"].view(batch["input_text"].shape[0], -1)
            for i in range(key.shape[0]):
                key_pad = key[i, :] != absa_dataset.padding_idx
                key_list.append(key[i, :].mul(key_pad))
            key_input = pad_sequence(key_list, padding_value=0, batch_first=True)
            val_b_size, val_type_num, val_passage_seq_len = batch["passage_list"].shape
            Val_review_outputs = ExpansionNet.run(
                key_input.to(device),
                     batch["user"].to(device),
                     batch["business"].to(device),
                    batch["output_text"].to(device),
                    encoder1,
                    encoder2,
                    encoder3,
                    decoder,
                    encoder1_optimizer,
                    encoder2_optimizer,
                    encoder3_optimizer,
                    decoder_optimizer,
                    device=device,
                    max_length=50
                     )
            Val_review_pred = torch.argmax(Val_review_outputs,dim=-1)

            Val_review_label = batch["output_text"].to(device)
            Val_review_label = torch.cat([Val_review_label[:, 1:],
                                      torch.LongTensor(val_b_size, 1).fill_(0).to(device)], axis=-1)


            Val_OP_loss_vals = []
            if loss_func == "cross_entropy":

                #
                # weight_tensor = torch.FloatTensor(
                #     [label_weight[i] for i in batch["aspect_label"][:, i].numpy()]).unsqueeze(-1)

                # weight_tensor = weight_tensor.repeat(1, Val_passage_loss_list.shape[1]).to(device)
                # Val_passage_loss = Val_passage_loss_list.sum(axis=0).mean()
                # Val_passage_loss = torch.mul(Val_passage_loss_list,weight_tensor).sum(axis=0).mean()


                # Avg of sum of token loss (after ignoring padding tokens)


                Val_RE_loss_vals = review_criterion(Val_review_outputs.transpose(1, 2),
                                         Val_review_label)
                # Avg of sum of token loss (after ignoring padding tokens)
                # loss = loss_vals
                Val_RE_Loss = Val_RE_loss_vals.sum(axis=0).mean()


            elif loss_func == "label_smoothing":



                # weight_tensor = torch.FloatTensor(
                #     [label_weight[i] for i in batch["aspect_label"][:, i].numpy()]).unsqueeze(-1)
                #
                # weight_tensor = weight_tensor.repeat(1, Val_passage_loss_list.shape[1]).to(device)
                # Val_passage_loss = torch.mul(Val_passage_loss_list, weight_tensor).sum(-1).mean()




                Val_RE_Loss = review_criterion(Val_review_outputs, Val_review_label).sum(-1).mean()

            Val_loss = Val_RE_Loss

            valid_loss_val = Val_loss.data.item() * batch["aspect_label"].shape[0]
            valid_loss += valid_loss_val

            # if batch_idx % 1000 == 0:
            #     review_max_len = opt.review_max_len
            #     review_tgt = ExpansionNet.generate_all_beamsearch(
            #         key.to(device),
            #         batch["user"].to(device),
            #         batch["business"].to(device),
            #         encoder1,
            #         encoder3,
            #         decoder,
            #         review_max_len,
            #         opt=opt,
            #         vocab_size = tgt_voc_size,
            #         beam_size=3,
            #         no_repeat_ngram_size=0,
            #         device=device
            #     )
            #
            #     print("----------------------------Val_Text-----------------------------------")
            #     # print(review_input)
            #     Val_review_label = batch["output_text"].to(device)
            #
            #
            #     # print(Val_review_label.shape)
            #     print("True: {}".format(denumericalize(Val_review_label, absa_dataset.tokenizer_out.idx2word)[0]))
            #     # print(opinion_tgt.shape)
            #     print("Pred: {}".format(denumericalize(review_tgt,
            #                                            absa_dataset.tokenizer_out.idx2word)[0]))
            #
            #     print("aspect_type:{}   Label:{}".format(type_label(batch["aspect_type"].data,
            #                                                         absa_dataset.tokenizer_type.idx2word)[0],
            #                                              type_label(batch["aspect_label"].data,
            #                                                         absa_dataset.tokenizer_aspect_label.idx2dependency)[
            #                                                  0]))




        valid_loss /= batch_idx
        print("\n")

        print('---------------------------------------------Epoch: {}, Training loss: {:.2f}, Valid loss: {:.2f}---------------------------'.format(
            epoch, training_loss, valid_loss))
        print("\n")
        log_data_list.append([epoch,
                              training_loss,
                              valid_loss])

        # TODO(Yoshi): The last model is redundant
        if epoch==0:
            f_out = codecs.open('log/' + opt.model_name + '_' + '_val.txt', 'a+', encoding="utf-8")
            f_out.write('time:{0}\n'.format(time_str))
            f_out.write(hyper_setting_name)
        torch.save(encoder1.state_dict(),
                   model_filepath.replace(".pt", "{}_{}_epoch-{}.pt".format(opt.model_name,"encoder1",epoch)))
        torch.save(encoder2.state_dict(),
                   model_filepath.replace(".pt", "{}_{}_epoch-{}.pt".format(opt.model_name,"encoder2", epoch)))
        torch.save(encoder3.state_dict(),
                   model_filepath.replace(".pt", "{}_{}_epoch-{}.pt".format(opt.model_name,"encoder3", epoch)))
        torch.save(decoder.state_dict(),
                   model_filepath.replace(".pt", "{}_{}_epoch-{}.pt".format(opt.model_name,"decoder", epoch)))

    torch.save(encoder1.state_dict(),
               model_filepath.replace(".pt", "{}_{}.pt".format(opt.model_name, "encoder1",)))
    torch.save(encoder2.state_dict(),
               model_filepath.replace(".pt", "{}_{}.pt".format(opt.model_name, "encoder2")))
    torch.save(encoder3.state_dict(),
               model_filepath.replace(".pt", "{}_{}.pt".format(opt.model_name, "encoder3")))
    torch.save(decoder.state_dict(),
               model_filepath.replace(".pt", "{}_{}.pt".format(opt.model_name, "decoder")))

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
