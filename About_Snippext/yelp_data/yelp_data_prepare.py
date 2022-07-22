##数据处理
import  pandas as pd
import  csv
import  json
def min_user_or_business(list_,min_review):
    min_dict = {}
    for key,value in list_:
        if value>=min_review:
            min_dict[key] = value
    return  min_dict






def writer_to_csv(out_filename,df_file,output_dict_user,output_dict_business,min_user_re,min_bus_re ,max_lenth=400):

    """
    :param out_filename: 输出的文件名字
    :param df_file:  dataframe 输入
    :param output_dict_user:  user_json文件
    :param output_dict_business:  business_json 文件
    :param min_user_re:   user的最少评论条数
    :param min_bus_re:   business 的最少被评论条数
    :param max_lenth: 最长的生成句子
    :return:
    """
    dict_user = {}
    dict_bus ={}

    #除去掉长度过长的
    df_file = df_file[df_file["text"].apply(lambda t:  isinstance(t,str))]
    try :
        df_file = df_file[df_file["text"].apply(lambda t: (len(t.split())) <max_lenth)]
    except:
        print("有错误")
    print("------------max_len remove------------------")
    #统计得到user情况
    for index,row in df_file.iterrows():
        user_id = row["user_id"]
        business_id= row["business_id"]
        row["text"] = row["text"].strip("\"")
        row["text"] = row["text"].strip("\'")
        if user_id not in dict_user:
            dict_user[user_id]=1
        else:
            dict_user[user_id]+=1
        if business_id not in dict_bus:
            dict_bus[business_id] = 1
        else:
            dict_bus[business_id] += 1
    dict_user = sorted(dict_user.items(), key=lambda x: x[1], reverse=True)
    dict_user = min_user_or_business(dict_user,min_user_re)
    dict_bus = sorted(dict_bus.items(), key=lambda x: x[1], reverse=True)
    dict_bus = min_user_or_business(dict_bus, min_bus_re)
    df_file = df_file[(df_file["user_id"].apply(lambda t:  t in dict_user) &(df_file["business_id"].apply(lambda t: t in dict_bus)))]
    print("------------user remove------------------")

    dict_user = {}
    dict_bus = {}
    for index, row in df_file.iterrows():
        user_id = row["user_id"]
        business_id = row["business_id"]
        if user_id not in dict_user:
            dict_user[user_id] = 1
        else:
            dict_user[user_id] += 1
        if business_id not in dict_bus:
            dict_bus[business_id] = 1
        else:
            dict_bus[business_id] += 1
    dict_user = sorted(dict_user.items(), key=lambda x: x[1], reverse=True)
    dict_user = min_user_or_business(dict_user, 0)
    dict_bus = sorted(dict_bus.items(), key=lambda x: x[1], reverse=True)
    dict_bus = min_user_or_business(dict_bus, 0)
    # for index, row in df_file.iterrows():
    #     business_id = row["business_id"]
    #     if business_id not in dict_bus:
    #         dict_bus[business_id] = 1
    #     else:
    #         dict_bus[business_id] += 1
    # dict_bus = sorted(dict_bus.items(), key=lambda x: x[1], reverse=True)
    # dict_bus  = min_user_or_business(dict_bus , min_bus_re)
    #
    # df_file = df_file[df_file["business_id"].apply(lambda t: t in dict_bus)]
    # print(df_file.shape[0])
    print("------------business get-----------------")
    # dict_user_new = {}
    # for index,row in df_file.iterrows():
    #     user_id = row["user_id"]
    #     if user_id not in dict_user_new:
    #         dict_user_new[user_id]=1
    #     else:
    #         dict_user_new[user_id]+=1
    #
    # dict_user_new = sorted(dict_user_new.items(), key=lambda x: x[1], reverse=True)
    # dict_user_new = min_user_or_business(dict_user_new,0)
    # print(dict_bus)
    #b保存user_review_dict
    with open(output_dict_user, 'w') as f:
        json.dump(dict_user, f)
    with open(output_dict_business, 'w') as f:
        json.dump(dict_bus, f)



    # df_file = df_file[df_file["business_id"].apply(lambda t: t in dict_bus)]

    #
    df_file=df_file.sort_values(by=['user_id'], na_position="first")

    print("-------------满足条件的行数：{0}---------------".format(df_file.shape[0]))
    print("-------------满足条件的user数量：{0}---------------".format(len(dict_user)))
    print("-------------business数量：{0}---------------".format(len(dict_bus)))

    # column_names = ["user_id", "business_id", "stars", "text"]
    df_file.to_csv(path_or_buf=out_filename,columns=["user_id","business_id","stars","text"],index=False)


if __name__ == "__main__":
    # use_model = "ex"
    use_model = "all"
    filename = "yelp_academic_dataset_review.csv"
    min_user_re = 8  # review最少评论的条数
    min_bus_re = 8  # business最少评论的条数
    if use_model == "ex":
        out_filename = "./example/example_yelp_aspect_datasets.csv"
        output_dict_user = "./example/example_user_dict.json"
        output_dict_business = "./example/example_business_dict.json"
        df_file = pd.read_csv(filename, sep=',', header=0, nrows=400000, encoding="utf-8")
    else:
        out_filename = "yelp_aspect_datasets.csv"
        output_dict_user = "user_dict.json"
        output_dict_business = "business_dict.json"
        df_file = pd.read_csv(filename, sep=',', header=0,nrows=60000000, encoding="utf-8")
    print("--------------total number:{0}----------".format(df_file.shape[0]))
    df_file = df_file.sort_values(by=['user_id'],na_position="first")
    writer_to_csv(out_filename, df_file,output_dict_user,output_dict_business,min_user_re,min_bus_re,max_lenth=200)







