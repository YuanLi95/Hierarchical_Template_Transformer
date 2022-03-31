import  json
import csv

fname = open("./data/yelp-default/test_old.json","r",encoding="utf-8")
# fnam_1 = open("./data/yelp-default/train.json","r",encoding="utf-8")
# fname_2 = open("./data/yelp-default/dev.json","r",encoding="utf-8")


csvFile = open("./my_out/output/TTH_reslut_example.csv", "r", encoding="utf-8")
csvFile_out = open("../../../Snippext_public-master/out_data/HTT_result_5000.csv", "w", encoding="utf-8", newline="")
reader = csv.reader(csvFile)
writer = csv.writer(csvFile_out)
new_test_list = []

fins = json.load(fname)
j=0

for i,item in enumerate(reader):
    if i==0:
        writer.writerow(item)
    for fin in fins:
        now_review  = fin["review"]
        if now_review.replace(" ","")==(item[1]).replace(" ",""):
            new_test_list.append(fin)
            j+=1
            writer.writerow(item)
            break
    print("this is i:{0} ".format(i))
    print(j)
b = json.dumps(new_test_list)
# print(new_test_list)
f2 = open('./data/yelp-default/test.json', 'w',encoding="utf-8")
f2.write(b)
f2.close()



