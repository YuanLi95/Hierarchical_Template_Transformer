import pandas as pd
import  json


#
# def get_need_number(pre_aspect_dict,ref_aspect_dict):
#     for asepct

def sentiment_label(sentiment):
    if sentiment =="positive":
        return 2
    elif sentiment == "neutral":
        return 1
    elif sentiment =="negative":
        return 0
    else:
        print("错误")
        exit()




if __name__ =="__main__":
    """
    aspect acc = aspect_true_number/aspect_all_number 
    aspect_sentiment_acc = aspect_sentiment_true_number/aspect_true_number  
    """
    pred_path = "aspect_key_words/HTT_aspect_out.json"
    ref_path = "aspect_key_words/test.json"
    aspect_all_number  = 0.0
    aspect_true_number = 0.0
    aspect_sentiment_true_number = 0.0
    pred_dicts = []
    for line in open(pred_path, 'r',encoding="utf-8"):
        pred_dicts.append(json.loads(line))
    with open(ref_path, 'r', encoding='utf-8') as f:
        ref_dicts = json.load(f)
    for idx in range(len(pred_dicts)):
        pred_dict  = pred_dicts[idx]
        aspect_team = []
        sentiment = []
        for extraction in pred_dict["extractions"]:
            aspect_team.append(extraction["aspect"])
            sentiment.append(sentiment_label(extraction["sentiment"]))
        print(aspect_team)
        print(sentiment)

        ref_dict = ref_dicts[idx]

        ref_aspect_list = ref_dict["aspect_word"]
        ref_sentiment_list = ref_dict["aspect_label"]
        print(ref_aspect_list)
        print(ref_sentiment_list)

        aspect_all_number+= len(ref_aspect_list)
        print(aspect_all_number)

        for i, x in enumerate(ref_aspect_list):
            aspect_true_list = []
            aspect_flag= 0
            for pre_aspect_index in range(len(aspect_team)):
                if x.lower() in (aspect_team[pre_aspect_index].lower()) or (aspect_team[pre_aspect_index].lower()) in  x.lower():
                    aspect_true_list.append(pre_aspect_index)
                    aspect_flag+=1
            if aspect_flag==0:
                continue
            else:
                print(aspect_true_list)
                aspect_true_number+=1
                flag = 0
                for index in aspect_true_list:
                    if str(sentiment[index])==str(ref_sentiment_list[i]):
                        flag+=1
                if flag!=0:
                    aspect_sentiment_true_number+=1
        print(aspect_true_number)
        print(aspect_sentiment_true_number)
    print("aspect acc:{0}".format(aspect_true_number/aspect_all_number))
    print(" aspect_sentiment_acc :{0}".format(aspect_sentiment_true_number/aspect_true_number))






