# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import  torch
import  json
import re
import scipy.sparse as sp
import spacy
nlp = spacy.load('en_core_web_sm')
def clean_en_text(text):
    # keep English, digital and space
    comp = re.compile('[^A-Z^a-z^0-9^ ^]')
    return comp.sub('', text)



def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            try:
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
            except:
                print('WARNING: corrupted word vector of {} when being loaded from GloVe.'.format(tokens[0]))
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, type_name):

    embedding_matrix_file_name = './embedding/{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type_name )

    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors ...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(-1 / np.sqrt(embed_dim), 1 / np.sqrt(embed_dim), (1, embed_dim))
        fname = './GloVe/glove.840B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix


def build_other_matrix(other2idx,dim, type_name):
    embedding_matrix_file_name = './embedding/{0}_{1}_embedding_matrix.pkl'.format(str(dim),type_name)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading edge vectors ...')
        embedding_matrix = np.zeros((len(other2idx), dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[2:, :] = np.random.uniform(-1 / np.sqrt(dim), 1 / np.sqrt(dim), (1, dim))
        # embedding_matrix[1, :] = np.random.uniform(-1, 0.25, (1, dependency_dim))

        print('building edge_matrix:', embedding_matrix_file_name)
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix




def build_word2pos(other2idx,dim=1, type_name=None):
    embedding_matrix_file_name = './embedding/{0}_{1}_embedding_matrix.pkl'.format(str(dim),type_name)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading edge vectors ...')
        embedding_matrix = np.zeros((len(other2idx), dim))  # idx 0 and 1 are all-zeros
        print()
        for i in range(len(other2idx)):
            embedding_matrix[i, :] = other2idx[i]


        print('building edge_matrix:', embedding_matrix_file_name)
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix



def build_position_matrix(position2idx, position_dim, type):
    embedding_matrix_file_name = '{0}_{1}_position_matrix.pkl'.format(str(position_dim), type)

    embedding_matrix = np.zeros((len(position2idx), position_dim))  # idx 0 and 1 are all-zeros
    # embedding_matrix[1, :] = np.random.uniform(-1 / np.sqrt(position_dim), 1 / np.sqrt(position_dim), (1, position_dim))
    embedding_matrix[1, :] = np.random.uniform(-0.25, 0.25, (1, position_dim))


    print('building position_matrix:', embedding_matrix_file_name)
    pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix


class Dependecynizer(object):
    def __init__(self, dependency2idx=None):
        if dependency2idx is None:
            self.dependency2idx = {}
            self.idx2dependency = {}
            self.idx2dependency_number={}
            self.idx = 0
            self.dependency2idx['<pad>'] = self.idx
            self.idx2dependency[self.idx] = '<pad>'
            self.idx2dependency_number['<pad>']=1
            self.idx += 1
            self.dependency2idx['<unk>'] = self.idx
            self.idx2dependency[self.idx] = '<unk>'
            self.idx2dependency_number['<unk>'] = 1
            self.idx += 1
        else:
            self.dependency2idx = dependency2idx
            self.idx2dependency = {v: k for k, v in dependency2idx.items()}
            self.idx2dependency_number = {v: k for k, v in dependency2idx.items()}
        self.idx2dependency_number = {}
    def fit_on_dependency(self, dependency_edge):
        dependency_edges = dependency_edge.lower()
        dependency_edges = dependency_edges.split()
        for dependency_edge in dependency_edges:
            if dependency_edge not in self.dependency2idx:
                self.dependency2idx[dependency_edge] = self.idx
                self.idx2dependency[self.idx] = dependency_edge
                self.idx2dependency_number[dependency_edge]=1
                self.idx += 1
            else:
                self.idx2dependency_number[dependency_edge] += 1
    def dependency_to_index(self,other_word):

        unknownidx = 1
        sequence = [self.dependency2idx[w] if w in self.dependency2idx else unknownidx for w in other_word]

        return  sequence[0]


class Positionnizer(object):
    def __init__(self, position2idx=None):
        if position2idx is None:
            self.position2idx = {}
            self.idx2position = {}
            self.idx = 0
            self.position2idx['<pad>'] = self.idx
            self.idx2position[self.idx] = '<pad>'
            self.idx += 1
            self.position2idx['<unk>'] = self.idx
            self.idx2position[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.position2idx = position2idx
            self.idx2position = {v: k for k, v in position2idx.items()}

    def fit_on_position(self, syntax_positions):
        for syntax_position in syntax_positions:
            if syntax_position not in self.position2idx:
                self.position2idx[syntax_position] = self.idx
                self.idx2position[self.idx] = syntax_position

                self.idx += 1
    def position_to_index(self,position_sequence):
        position_sequence = position_sequence.astype(np.str)
        unknownidx = 1
        position_matrix = [self.position2idx[w] if w in self.position2idx else unknownidx for w in position_sequence]
        return position_matrix


class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
            self.word2idx['<eos>'] = self.idx
            self.idx2word[self.idx] = '<eos>'
            self.idx += 1
            self.word2idx['<bos>'] = self.idx
            self.idx2word[self.idx] = '<bos>'
            self.idx += 1
            self.word2idx['<sep>'] = self.idx
            self.idx2word[self.idx] = '<sep>'
            self.idx += 1

        else:
            self.word2idx = word2idx
            self.idx2word = {v: k for k, v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
        text = text.lower()
        words = text.split()
        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return sequence


class ABSADataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ABSADatesetReader:

    @staticmethod
    def __read_text__(fnames):
        in_text = ''
        out_text = ''
        user_id = ""
        bus_id = ""
        stars_all = ""
        aspect_type_all = ""
        aspect_label = ""
        aspect_POS = ""
        passage_list = ""
        ratting = ""
        for fname in fnames:
            with open(fname, 'r') as f:
                fins = json.load(f)
            for fin in fins:
                # print(fin)
                rating_value = 0
                try:
                    out_text += (fin["review"]).lower()
                except:
                    print(fin)
                    print(out_text)
                in_text += (" "+" ".join(fin["aspect_word"])).lower()
                user_id +=" "+fin["user_id"]
                bus_id += " "+fin["business_id"]
                stars_all+=" "+str(fin["stars"])
                aspect_type_all+=" "+" ".join(fin["aspect_type"])
                passage_list+= " ".join(fin["passage_list"]).lower()

                for label in fin["aspect_label"]:
                    aspect_label += " "+ str(label)
                    rating_value += int(label)
                aspect_POS+=" ".join(fin["aspect_POS"])
                ratting += " "+str(rating_value)+" "


        return in_text,out_text,user_id,bus_id,stars_all,aspect_type_all,aspect_label,passage_list,aspect_POS,ratting

    @staticmethod
    def __read_data__(fname, tokenizer_type, tokenizer_input,tokenizer_out,tokenizer_user,tokenizer_business,tokenizer_aspect_label,tokenizer_passage,tokenizer_aspect_pos,out2pos,tokenizer_rating_label):
        f = open(fname, 'r')
        fins = json.load(f)
        all_data = []
        total_length = 0
        for i in range(0, len(fins)):
            row_all = fins[i]
            # print(row_all)
            # print(row_all)


            output_text_indices =  tokenizer_out.text_to_sequence("<bos>"+" "+row_all["review"]+" "+"<eos>")
            # print(output_text_indices)
            total_length += (len(output_text_indices)-2)
            # print(row_all["user_id"])
            user_indices = tokenizer_user.text_to_sequence(row_all["user_id"])
            business_indices = tokenizer_business.text_to_sequence(row_all["business_id"])

            pos_indices = tokenizer_aspect_pos.text_to_sequence("<bos>"+" "+" ".join(row_all["aspect_POS"])+" "+"<eos>")
            output_pos_indices =  [out2pos[i] for i in output_text_indices]

            rating_value = row_all["stars"]


            # print(len(pos_indices))
            # print(len(output_text_indices))
            # exit()
            input_text_indices = []
            passage_list_indices = []
            aspect_label_indices  =[]
            aspect_type_indices = []

            passage_inputs_i = "<bos>"

            for j  in  range(len(row_all["aspect_label"])):

                input_text_indices.append(tokenizer_input.text_to_sequence(row_all["aspect_word"][j]))
                passage_list_indices.append(tokenizer_passage.text_to_sequence("<bos>"+" "+row_all["passage_list"][j]+" "+"<eos>"))

                passage_inputs_i+=" "+row_all["passage_list"][j]+" "
                aspect_label_indices.append(tokenizer_aspect_label.dependency_to_index(row_all["aspect_label"][j]))
                aspect_type_indices.append(tokenizer_type.text_to_sequence(row_all["aspect_type"][j]))

                if j!=(len(row_all["aspect_label"])-1):
                    passage_inputs_i+="<sep>"
                else:
                    passage_inputs_i += "<eos>"
            rating_indices = tokenizer_rating_label.dependency_to_index(str(rating_value))
            passage_input_indices=tokenizer_passage.text_to_sequence(passage_inputs_i)

            # print(passage_input_indices)

            data = {
                "aspect_type":aspect_type_indices,
                'input_text': input_text_indices,
                'passage_list': passage_list_indices,
                'passage_input':passage_input_indices,
                'output_text': output_text_indices,
                'aspect_label': aspect_label_indices,
                'user':user_indices,
                'business':business_indices,
                'temp_pos':pos_indices,
                'output_pos':output_pos_indices,
                "rating":rating_indices,
            }
            all_data.append(data)

        return total_length,all_data

    def __init__(self, embed_dim,user_embed_dim,bus_embed_dim,pos_embed_dim,label_embed_dim,aspect_type_dim,basepath,data_sets = "yelp"):
        print("preparing {0} dataset ...".format(data_sets))
        data_dirpath = os.path.join(basepath,
                                    "./data/yelp-default/example/"
                                    )
        data_dirpath = data_dirpath.replace("\\", "/")

        train_filepath = os.path.join(data_dirpath,
                                      "train.json").replace("\\", "/")
        valid_filepath = os.path.join(data_dirpath,
                                      "dev.json").replace("\\", "/")
        test_filepath = os.path.join(data_dirpath,
                                      "test.json").replace("\\", "/")
        print(train_filepath)
        fname = {
            'yelp': {
                'train': train_filepath,
                'dev' : valid_filepath,
                'test': test_filepath
            }

        }
        in_text, out_text, user_id, bus_id, stars_all, aspect_type_all,aspect_label,passage,aspect_POS,ratting = ABSADatesetReader.__read_text__([fname[data_sets]['train'], fname[data_sets]['dev'],fname[data_sets]["test"]])
        #构建aspect_type 词表
        if os.path.exists("./all2idx/aspect_type" + '_aspect2idx.pkl'):
            with open("./all2idx/aspect_type" + '_aspect2idx.pkl', 'rb') as f:
                word2idx_1 = pickle.load(f)
                tokenizer_type = Tokenizer(word2idx=word2idx_1)
        else:
            tokenizer_type = Tokenizer()
            tokenizer_type.fit_on_text(aspect_type_all)
            with open("./all2idx/aspect_type"+ '_word2idx.pkl', 'wb') as f:
                pickle.dump(tokenizer_type.word2idx, f)


        # # 构建in_text 的词表
        if os.path.exists("./all2idx/in_text" + '_word2idx.pkl'):
            with open("./all2idx/in_text" + '_word2idx.pkl', 'rb') as f:
                word2idx_2 = pickle.load(f)
                tokenizer_input = Tokenizer(word2idx=word2idx_2)
        else:
            tokenizer_input = Tokenizer()
            tokenizer_input.fit_on_text(in_text)
            with open("./all2idx/in_text"+ '_word2idx.pkl', 'wb') as f:
                pickle.dump(tokenizer_input.word2idx, f)


        #
        # #构建out_text 的词典
        #
        if os.path.exists("./all2idx/out_text" + '_word2idx.pkl'):
            with open("./all2idx/out_text" + '_word2idx.pkl', 'rb') as f:
                word2idx_3 = pickle.load(f)
                tokenizer_out = Tokenizer(word2idx=word2idx_3)
        else:
            tokenizer_out = Tokenizer()
            tokenizer_out.fit_on_text(out_text)
            with open("./all2idx/out_text"+ '_word2idx.pkl', 'wb') as f:
                pickle.dump(tokenizer_out.word2idx, f)



        #构建user_id 的词典
        if os.path.exists("./all2idx/user" + '_user2idx.pkl'):
            with open("./all2idx/user" + '_user2idx.pkl', 'rb') as f:
                user2index = pickle.load(f)
                tokenizer_user= Tokenizer(word2idx=user2index)
                #
        else:
            tokenizer_user= Tokenizer()
            tokenizer_user.fit_on_text(user_id)
            with open("./all2idx/user" + '_user2idx.pkl', 'wb') as f:
                pickle.dump(tokenizer_user.word2idx,f)
                print(tokenizer_user.word2idx)


        #构建business_id 词典
        if os.path.exists("./all2idx/business" + '_user2idx.pkl'):
            with open("./all2idx/business" + '_user2idx.pkl', 'rb') as f:
                business2index = pickle.load(f)
                tokenizer_business= Tokenizer(word2idx=business2index)
                #
        else:
            tokenizer_business= Tokenizer()
            tokenizer_business.fit_on_text(bus_id)
            with open("./all2idx/business" + '_user2idx.pkl', 'wb') as f:
                pickle.dump(tokenizer_business.word2idx,f)
                print(tokenizer_business.word2idx)

        # 构建  sentiment_label_embedding
        if os.path.exists("./all2idx/sentiment" + '_label2idx.pkl'):
            with open("./all2idx/sentiment" + '_label2idx.pkl', 'rb') as f:
                sentiment2idex = pickle.load(f)
                tokenizer_aspect_label = Dependecynizer(dependency2idx=sentiment2idex)
                #
        else:
            tokenizer_aspect_label = Dependecynizer()
            tokenizer_aspect_label.fit_on_dependency(aspect_label)
            with open("./all2idx/sentiment" + '_label2idx.pkl', 'wb') as f:
                pickle.dump(tokenizer_aspect_label.dependency2idx, f)

        if  os.path.exists("./all2idx/rating" + '_label2idx.pkl'):
            with open("./all2idx/sentiment" + '_label2idx.pkl', 'rb') as f:
                ratting2idex = pickle.load(f)
                tokenizer_rating_label = Dependecynizer(dependency2idx=ratting2idex)
                #
        else:
            tokenizer_rating_label = Dependecynizer()
            tokenizer_rating_label.fit_on_dependency(stars_all)
            with open("./all2idx/rating" + '_label2idx.pkl', 'wb') as f:
                pickle.dump(tokenizer_rating_label.dependency2idx, f)

        self.pos_label_idx = tokenizer_aspect_label.dependency2idx["2"]
        self.neu_label_idx = tokenizer_aspect_label.dependency2idx["1"]
        self.neg_label_idx = tokenizer_aspect_label.dependency2idx["0"]
        print(tokenizer_aspect_label.dependency2idx)
        # 构建Passage_list embedding
        if os.path.exists("./all2idx/passage" + '_word2idx.pkl'):
            with open("./all2idx/passage" + '_word2idx.pkl', 'rb') as f:
                word2idx_4 = pickle.load(f)
                tokenizer_passage = Tokenizer(word2idx=word2idx_4)
        else:
            tokenizer_passage = Tokenizer()
            tokenizer_passage.fit_on_text(passage)
            with open("./all2idx/passage"+ '_word2idx.pkl', 'wb') as f:
                pickle.dump(tokenizer_passage.word2idx, f)


        # 构建POS_embedding
                # 构建  sentiment_label_embedding
        if os.path.exists("./all2idx/pos2idx.pkl"):
            with open("./all2idx/pos2idx.pkl", 'rb') as f:
                pos2idex = pickle.load(f)
                tokenizer_aspect_pos = Tokenizer(word2idx=pos2idex)
                #
        else:
            tokenizer_aspect_pos = Tokenizer()
            tokenizer_aspect_pos.fit_on_text(aspect_POS)
            with open("./all2idx/pos2idx.pkl", 'wb') as f:
                pickle.dump(tokenizer_aspect_pos.word2idx, f)

        # out_to_pos
        if os.path.exists("./all2idx/out2pos.pkl"):
            with open("./all2idx/out2pos.pkl", 'rb') as f:
                self.out2pos =  pickle.load(f)
        else:
            with open("./all2idx/out2pos.pkl", 'wb') as f:
                word_to_pos_indx = {}
                word_to_pos_indx[0] = 0
                word_to_pos_indx[1] = 1
                word_to_pos_indx[2] = 2
                word_to_pos_indx[3] = 3
                i = 0
                for (key, value) in tokenizer_out.word2idx.items():
                    if i >= 4:
                        word_pos = nlp(key)
                        for token in word_pos:
                            word_to_pos_indx[value] = tokenizer_aspect_pos.word2idx[token.pos_.lower()]
                    # print(key,value)
                    i = i + 1
                    print(i)
                self.out2pos = word_to_pos_indx

                pickle.dump(word_to_pos_indx, f)









        # print(tokenizer_business.idx2dependency_number)
        # print(tokenizer_business.dependency2idx)
        # print(tokenizer_user.dependency2idx)
        # print(tokenizer_type.word2idx)
        # print(tokenizer_aspect_label.dependency2idx)
        self.tokenizer_out = tokenizer_out
        self.tokenizer_type = tokenizer_type
        self.tokenizer_aspect_label =tokenizer_aspect_label
        self.tokenizer_passage = tokenizer_passage
        self.tokenizer_input= tokenizer_input
        self.padding_idx = tokenizer_input.word2idx["<pad>"]
        self.bos_idx = tokenizer_input.word2idx["<bos>"]
        self.eos_idx = tokenizer_input.word2idx["<eos>"]
        self.unk_idx = tokenizer_input.word2idx["<unk>"]

        self.aspect_type_embedding = build_other_matrix(tokenizer_type.word2idx,aspect_type_dim,"aspect_type")

        self.input_text_embedding = build_embedding_matrix(tokenizer_input.word2idx, embed_dim, "input_text")

        self.pass_list_embedding = build_embedding_matrix(tokenizer_passage.word2idx,embed_dim,"passage")
        self.out_text_embedding = build_embedding_matrix(tokenizer_out.word2idx, embed_dim, "out_text")

        self.user_embedding = build_other_matrix(tokenizer_user.word2idx,user_embed_dim,"user_id")

        self.business_embedding = build_other_matrix(tokenizer_business.word2idx,bus_embed_dim,"business_id")

        self.label_embedding = build_other_matrix(tokenizer_aspect_label.dependency2idx,label_embed_dim,"aspect_label")

        self.pos_embedding = build_other_matrix(tokenizer_aspect_pos.word2idx,pos_embed_dim,"text_pos")

        self.out2pos_index = build_word2pos(self.out2pos,dim=1,type_name="word2pos")

        self.rating =build_other_matrix(tokenizer_rating_label.dependency2idx,56,"rating")

        print("user_number:{0}".format(self.user_embedding.shape[0]))
        print("product_number:{0}".format(self.business_embedding.shape[0]))



        # print(self.aspect_type_embedding.shape)
        # print(self.input_text_embedding.shape)
        # print(self.out_text_embedding.shape)
        # print(self.user_embedding.shape)
        # print(self.business_embedding.shape)
        # print(self.label_embedding.shape)
        # print(tokenizer_user.word2idx)

        total_length_train,self.train_data = ABSADataset(ABSADatesetReader.__read_data__(fname[data_sets]['train'],tokenizer_type, tokenizer_input,tokenizer_out,tokenizer_user,tokenizer_business,tokenizer_aspect_label,tokenizer_passage,tokenizer_aspect_pos,self.out2pos,tokenizer_rating_label))

        total_length_dev,self.dev_data = ABSADataset(ABSADatesetReader.__read_data__(fname[data_sets]['dev'],tokenizer_type, tokenizer_input,tokenizer_out,tokenizer_user,tokenizer_business,tokenizer_aspect_label,tokenizer_passage,tokenizer_aspect_pos,self.out2pos,tokenizer_rating_label))

        total_length_test,self.test_data = ABSADataset(ABSADatesetReader.__read_data__(fname[data_sets]['test'], tokenizer_type, tokenizer_input,tokenizer_out,tokenizer_user,tokenizer_business,tokenizer_aspect_label,tokenizer_passage,tokenizer_aspect_pos,self.out2pos,tokenizer_rating_label))

        print("the mean of train text length: {0} ".format(total_length_train/len(self.train_data)))
        # print(len(self.train_data))
        print("the mean of dev text length：{0}".format(total_length_dev/ (len(self.dev_data))))
        # print(len(self.train_data))
        print("the mean of dev test length：{0}".format(total_length_test/ len(self.test_data)))
        # exit()