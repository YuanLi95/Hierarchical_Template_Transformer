# -*- coding: utf-8 -*-

import math
import random
import torch
import numpy
import  numpy as np

class BucketIterator(object):
    def __init__(self, data, batch_size, sort_key='aspect_label', shuffle=True, sort=True,max_len=70):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.out_text_key = "output_text"
        self.input_key = "input_text"
        self.passage_list_key = "passage_list"
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batches

    def pad_data(self, batch_data):


        batch_aspect_type =[]
        batch_input_text = []
        batch_passage_list = []
        batch_passage_input = []
        batch_output_text = []
        batch_output_pos = []
        batch_aspect_label = []
        batch_temp = []

        # for  t in batch_data:
        #     print(t)
        aspect_number_i = []
        out_len_i =[]
        temp_len_i = []
        passage_input_i=[]
        for t in batch_data:
            aspect_number_i.append(len(t[self.sort_key]))
            out_len_i.append(len(t[self.out_text_key]))
            temp_len_i.append(len(t["temp_pos"]))
            passage_input_i.append(len(t["passage_input"]))
        aspect_number_max= max(aspect_number_i)
        out_max_len = max(out_len_i)
        temp_max_len = max(temp_len_i)
        passage_input_max_len=max(passage_input_i)
        batch_passage_max = []
        batch_input_max= []
        for t in batch_data:
            for i in t[self.passage_list_key]:
                batch_passage_max.append(len(i))
            for j in t[self.input_key]:
                batch_input_max .append(len(j))

        batch_passage_max = np.max(batch_passage_max)
        batch_input_max = np.max(batch_input_max)

        # print(aspect_number_max)

        batch_user = [t["user"][0] for t in batch_data]
        batch_business = [t["business"][0] for t in batch_data]

        batch_rating = [t["rating"] for t in batch_data]
        for item in batch_data:
            # print(item)

            aspect_type_indices, input_text_indices, passage_list_indices,passage_input_indices, output_text_indices,\
            aspect_label_indices,aspect_user_indices,aspect_business_indices,temp_indices,output_pos_indices,= \
                item['aspect_type'], item['input_text'], item['passage_list'],item['passage_input'], item['output_text'],\
                item['aspect_label'], item['user'],item['business'],item['temp_pos'],item["output_pos"],
            aspect_number_padding = [0] * (aspect_number_max - len(aspect_type_indices))
            output_text_padding = [0]*(out_max_len- len(output_text_indices))

            temp_padding = [0]*(temp_max_len-len(temp_indices))
            passage_input_padding = [0]*(passage_input_max_len-len(passage_input_indices))


            batch_aspect_label.append(aspect_label_indices+aspect_number_padding)
            batch_passage_input.append(passage_input_indices+passage_input_padding)

            # batch_aspect_type.append(aspect_type_indices+aspect_number_padding)
            batch_aspect_type.append(([y for x in aspect_type_indices for y in x] +aspect_number_padding))

            batch_output_text.append(output_text_indices+output_text_padding)
            batch_output_pos.append(output_pos_indices + output_text_padding)

            batch_temp.append(temp_indices+temp_padding)

            #对内层进行pad
            for i in range(len(aspect_type_indices)):
                input_text_indices[i] = input_text_indices[i]+[0]*(batch_input_max-len(input_text_indices[i]))
                passage_list_indices[i] = passage_list_indices[i]+[0]*(batch_passage_max-len(passage_list_indices[i]))

            batch_input_text.append(numpy.pad(input_text_indices,
            ((0,aspect_number_max-len(aspect_type_indices)),(0,0)),'constant'))

            batch_passage_list.append(numpy.pad(passage_list_indices,
          ((0, aspect_number_max - len(aspect_type_indices)), (0, 0)), 'constant'))
        # batch_aspect_type = []
        # batch_input_text = []
        # batch_passage_list = []
        # batch_output_text = []
        # batch_aspect_label = []
        # batch_user = []
        # batch_business = []
        # print("-"*30)
        # print(batch_input_text)
        # print(batch_passage_list)
        # print(batch_output_text)
        # print(batch_business)
        # print(batch_aspect_label)
        # print(batch_aspect_type)
        # print(batch_user)
        # # exit()
        # # 填充
        # print(aspect_number_padding)
        # print(batch_aspect_type)
        # print(torch.tensor(batch_aspect_type))
        # print(torch.tensor(batch_input_text))
        # print(torch.tensor(batch_passage_list))
        # print(torch.tensor(batch_aspect_label))
        # print(torch.tensor(batch_user))
        # print(torch.tensor(batch_business))
        # print(batch_aspect_label)
        # print(batch_aspect_type)
        # print(torch.tensor(batch_aspect_label))
        # print(torch.tensor(batch_temp))
        # exit()'

        return {
                'aspect_type':torch.tensor(batch_aspect_type), \
                'input_text': torch.tensor(batch_input_text).long(), \
                'passage_list': torch.tensor(batch_passage_list).long(), \
                'review_input':torch.tensor(batch_passage_input).long(),\
                'output_text': torch.tensor(batch_output_text), \
                'aspect_label': torch.tensor(batch_aspect_label), \
                'user': torch.tensor(batch_user), \
                'business':torch.tensor(batch_business),\
                'temp_pos':torch.tensor(batch_temp),
                "output_pos":torch.tensor(batch_output_pos),
                "rating":torch.tensor(batch_rating)
            }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]
