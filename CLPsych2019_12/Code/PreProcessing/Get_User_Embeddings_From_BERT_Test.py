# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 23:15:57 2019

@author: AshwinAmbal
"""

import csv
from collections import defaultdict
just_user_posts_train = list()
with open("/home/aambalav/CLPsych_Challenge/Dataset/PreProcessing/Only_User_Posts_Test_Task_C.txt",'r', encoding = 'utf8', newline='') as outcsv:   
        reader = csv.reader(outcsv,quotechar = '"')
        for row in reader:
            just_user_posts_train.append(row)
            
import json
with open('/home/aambalav/CLPsych_Challenge/Dataset/PreProcessing/User_To_Posts_Test_Task_C.json', 'r') as fp:
    posts_users_individual = json.load(fp)
    
sent_to_idx = dict()

for i, row in enumerate(just_user_posts_train):
    sent_to_idx[row[0]] = i
    
idx_to_user = dict()
    
for user in posts_users_individual:
    for sent in posts_users_individual[user]:
        idx_to_user[sent_to_idx[sent]] = user

word_embed_sent = list()        
for line in open('/home/aambalav/CLPsych_Challenge/Dataset/PreProcessing/Train_Normal_Embed_Test_Task_C.json', 'r'):
    json_file = json.loads(line)
    word_embed_sent.append(list(list([float(v) for v in json_file['features'][0]['layers'][0]['values']])))

embedding_users_individual = defaultdict(list)  

for index, user in idx_to_user.items():
    embedding_users_individual[user].append(word_embed_sent[index])
    
with open('/home/aambalav/CLPsych_Challenge/Dataset/PreProcessing/User_To_Embeddings_Test_Task_C.json', 'w') as fp:
    json.dump(embedding_users_individual, fp)