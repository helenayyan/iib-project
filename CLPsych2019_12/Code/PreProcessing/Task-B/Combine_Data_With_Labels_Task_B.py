# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 12:50:46 2019

@author: AshwinAmbal
"""
import csv
path ="/home/yy452/rds/rds-gvdd-Yuap0gjVpKM/yy452/umd_reddit_suicidewatch_dataset_v2/crowd/test/crowd_test_C.csv"
file =open(path, 'r', encoding = 'utf8')
reader_data = csv.reader(file, delimiter=',')
user_label = list(reader_data)
all_user_id_label = dict()

for i, row in enumerate(user_label):
    if(i == 0):
        continue
    all_user_id_label[row[0]] = row[1]
    
path ="/home/yy452/rds/rds-gvdd-Yuap0gjVpKM/yy452/umd_reddit_suicidewatch_dataset_v2/crowd/test/task_C_test.posts.csv"
file = open(path, 'r', encoding = 'utf8')
reader_data = csv.reader(file, delimiter=',')

user_id_label = dict()

for i, row in enumerate(reader_data):
    if(i == 0):
        continue
    user_id_label[row[1]] = all_user_id_label[row[1]]
    

path ="/home/yy452/rds/rds-gvdd-Yuap0gjVpKM/yy452/CLPsych2019_12/Dataset/task_c/testUserIds_TaskC_Final.csv"
with open(path, 'w', encoding='utf8', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    for key, value in user_id_label.items():
        writer.writerow([key, value])