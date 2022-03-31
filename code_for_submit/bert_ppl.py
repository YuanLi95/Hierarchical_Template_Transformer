import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer,BertForMaskedLM
import pandas as pd
from pandas.core.frame import  DataFrame
# Load pre-trained model (weights)\
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0,1,2,3" if USE_CUDA else "cpu")
# device = torch.device("cpu")
with torch.no_grad():
    model = BertForMaskedLM.from_pretrained('bert-large-uncased').to(device)
    # print(model)
    model.eval()
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
def ppl_score(sentence):
    # print(sentence)
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)]).to(device=device)
    sentence_loss=0.
    for i,word in enumerate(tokenize_input):

        tokenize_input[i]='[MASK]'
        mask_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)]).to(device)
        word_loss=model(mask_input, masked_lm_labels=tensor_input).cpu().data.numpy()
        sentence_loss +=word_loss
        #print("Word: %s : %f"%(word, np.exp(-word_loss)))
    # print(sentence_loss)
    # print(np.exp(sentence_loss/len(tokenize_input)))
    return np.exp(sentence_loss/len(tokenize_input))


if __name__ == "__main__":
    import csv
    my_out_path = "./my_out/output/THH_out.csv"
    csvFile = open(my_out_path,"r",encoding="utf-8")
    reader = csv.reader(csvFile,)
    csvFile = open("./my_out/output/TTH_reslut_example.csv", "w", encoding="utf-8", newline ="")
    writer = csv.writer(csvFile,)

    for i,item in enumerate(reader):
        print(item)
        exit()
        if i ==0:
            item.append("ppl_score")
        item.append(ppl_score(item[0]))
        print(i)
        writer.writerow(item)

