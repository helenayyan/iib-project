"""
Write files in BERT Required Format
"""


import random
import sys
import csv

csv.field_size_limit(sys.maxsize)
req = list()
for i in range(1, 574):
    req.append(i)
req_final = random.sample(req,  57)
train_set = list()
dev_set = list()

with open("/home/yy452/rds/rds-gvdd-Yuap0gjVpKM/yy452/CLPsych2019_12/Dataset/task_c/Full_Train_Data_Final.tsv", 'r', encoding='utf8') as csv_file:
    reader = csv.reader(csv_file, delimiter="\t", quotechar=None)
    for i, row in enumerate(reader):
        if(i == 0):
            header = row
        elif(i in req_final):
            dev_set.append(row)
        else:
            train_set.append(row)

full_train_set = train_set + dev_set

with open("/home/yy452/rds/rds-gvdd-Yuap0gjVpKM/yy452/CLPsych2019_12/Dataset/task_c/User_Posts_Processed_Train_Final.tsv",'w', encoding = 'utf8', newline='') as outcsv:   
        writer = csv.writer(outcsv, delimiter="\t", quotechar=None)
        writer.writerow(header)
        for row in train_set:
            writer.writerow(row)
            
with open("/home/yy452/rds/rds-gvdd-Yuap0gjVpKM/yy452/CLPsych2019_12/Dataset/task_c/User_Posts_Processed_Dev_Final.tsv",'w', encoding = 'utf8', newline='') as outcsv:   
        writer = csv.writer(outcsv, delimiter="\t", quotechar=None)
        writer.writerow(header)
        for row in dev_set:
            writer.writerow(row)

with open("/home/yy452/rds/rds-gvdd-Yuap0gjVpKM/yy452/CLPsych2019_12/Dataset/task_c/User_Posts_Processed_Train_Full_Final.tsv",'w', encoding = 'utf8', newline='') as outcsv:   
        writer = csv.writer(outcsv, delimiter="\t", quotechar=None)
        writer.writerow(header)
        for row in full_train_set:
            writer.writerow(row)


# import pandas as pd
# df = pd.read_csv('C:\\CLPsych Challenge\\Dataset\\PreProcessing\\User_Posts_Processed_Test.csv')
# df.to_csv('C:\\CLPsych Challenge\\Dataset\\PreProcessing\\User_Posts_Processed_Test_Final.tsv', sep='\t', index=False, header=True)
# # if you are creating test.tsv, set header=True instead of False

#df = pd.read_csv('C:\\CLPsych Challenge\\Temp\\train.csv', header=None)
#df.columns = ["id", "label", "a", "sentence"]
#df = df.drop(["label", "a"], axis=1)
#df.to_csv('C:\\CLPsych Challenge\\Temp\\test.tsv', sep='\t', index=False, header=True)