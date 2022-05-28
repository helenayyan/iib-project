"""
Read CLS Embeddings from extracted BERT Embeddings to train MIL
"""

import json
import csv
import numpy as np

word_embed = list()
for line in open('C:\\CLPsych Challenge\\Train_Test_Embeddings\\Test_Normal_Embed.json', 'r'):
    json_file = json.loads(line)
    word_embed.append(list(list([float(v) for v in json_file['features'][0]['layers'][0]['values']])))

word_embed_sent = list()
for line in open('C:\\CLPsych Challenge\\Train_Test_Embeddings\\Test_Normal_Embed_Sentiment.json', 'r'):
    json_file = json.loads(line)
    word_embed_sent.append(list(list([float(v) for v in json_file['features'][0]['layers'][0]['values']])))

user_posts_final = list()
with open("C:\\CLPsych Challenge\\Dataset\\PreProcessing\\PreProcessed-Data\\User_Posts_Processed_Test_Final.tsv",'r', encoding = 'utf8', newline='') as outcsv:   
        reader = csv.reader(outcsv, delimiter='\t',quotechar = '"')
        for i, row in enumerate(reader):
            if(i == 0):
                continue
            user_posts_final.append(row)
user_to_embed_map = dict()
for user, norm_embed, senti_embed in zip(user_posts_final, word_embed, word_embed_sent):
    user_to_embed_map[user[0]] = [norm_embed, senti_embed]
        
    
"""
word_embed = list()
for line in open('C:\\CLPsych Challenge\\Train_Test_Embeddings\\Train_Normal_Embed.json', 'r'):
    json_file = json.loads(line)
    word_embed.append(list(list([float(v) for v in json_file['features'][0]['layers'][0]['values']])))

word_embed_sent = list()
for line in open('C:\\CLPsych Challenge\\Train_Test_Embeddings\\Train_Normal_Embed_Sentiment.json', 'r'):
    json_file = json.loads(line)
    word_embed_sent.append(list(list([float(v) for v in json_file['features'][0]['layers'][0]['values']])))

user_posts_final = list()
with open("C:\\CLPsych Challenge\\Dataset\\PreProcessing\\PreProcessed-Data\\User_Posts_Processed_Train_Full_Final.tsv",'r', encoding = 'utf8', newline='') as outcsv:   
        reader = csv.reader(outcsv, delimiter='\t',quotechar = '"')
        for i, row in enumerate(reader):
            if(i == 0):
                continue
            user_posts_final.append(row)
"""
for user, norm_embed, senti_embed in zip(user_posts_final, word_embed, word_embed_sent):
    user_to_embed_map[user[0]] = [norm_embed, senti_embed]

with open('C:\\CLPsych Challenge\\Dataset\\PreProcessing\\Final_Embed_User.json', 'w') as fp:
    json.dump(user_to_embed_map, fp)
