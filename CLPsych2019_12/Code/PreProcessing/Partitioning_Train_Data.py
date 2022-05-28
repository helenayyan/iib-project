# -*- coding: utf-8 -*-
"""
Splits the Full Train Data into Train-Test Sets and Oversampling the training data
"""
import csv
import random

#user_posts = list()
user_posts = list()

train_data = list()
test_data = list()

with open("C:\\CLPsych Challenge\\Dataset\\clpsych19_training_data\\trainUserIds_TaskA.csv") as outcsv:
    reader = csv.reader(outcsv, delimiter=',', quotechar='"')
    for i, row in enumerate(reader):
        if i == 0:
            continue
        train_data.append(row[0])

with open("C:\\CLPsych Challenge\\Dataset\\clpsych19_training_data\\testUserIds_TaskA.csv") as outcsv:
    reader = csv.reader(outcsv, delimiter=',', quotechar='"')
    for i, row in enumerate(reader):
        if i == 0:
            continue
        test_data.append(row)

with open("C:\\CLPsych Challenge\\Dataset\\PreProcessing\\Task A-BERT_FINE_TUNING\\Pre-Processed-Data-Normal\\User_Posts_Processed_Train_Full_Final_Normal.tsv",'r', encoding = 'utf8', newline='') as outcsv:   
    reader = csv.reader(outcsv, delimiter='\t', quotechar='"')
    for i, row in enumerate(reader):
        if i == 0:
            continue
        user_posts.append(row)

final_train_data = list()
final_train_data.append(['User ID', 'Post', 'Label'])
final_test_data = list()   
final_test_data.append(['User ID', 'Post', 'Label'])
for post in user_posts:
    if post[0] in train_data:
        final_train_data.append(post)
    else:
        final_test_data.append(post)

# typeOfSampling = Over or Under Sampling the Data. 0 for undersample, 1 for oversample)
# num_samples_to_change = Number of samples to add or remove
# class_name = the class you want to sample.            
def over_under_sample(posts, class_name, typeOfSampling, num_samples_to_change):
    edited_posts = []
    temp_posts = []
    req = []
    for row in posts:
        if(row[2] == class_name):
            temp_posts.append(row)
        else:
            edited_posts.append(row)
    random.seed(9001)
    if(typeOfSampling == 0):
        for i in range(0, len(temp_posts)):
            req.append(i)
        req_final = random.sample(req, len(temp_posts) - num_samples_to_change)
        for i in req_final:
            edited_posts.append(temp_posts[i])
    else:
        edited_posts.extend(temp_posts)
        while num_samples_to_change - len(temp_posts) > 0:
            edited_posts.extend(temp_posts)
            num_samples_to_change -= len(temp_posts)
        while(num_samples_to_change > 0):
            num = random.randint(0, len(temp_posts))
            edited_posts.append(temp_posts[num])
            num_samples_to_change -= 1
    return edited_posts
        
# user_posts_undersampled = over_under_sample(user_posts_final, 'd', 0, 149 - 85)

user_posts_oversampled = over_under_sample(final_train_data, 'b', 1, 84 - 29)
    

count_label_class = dict()
for row in user_posts_oversampled:
    if(row[2] in count_label_class):
        count_label_class[row[2]] += 1
    else:
        count_label_class[row[2]] = 1

with open("C:\\CLPsych Challenge\\Dataset\\PreProcessing\\Task A-BERT_FINE_TUNING\\Pre-Processed-Data-Normal\\User_Posts_Processed_Train_Full_Final.tsv",'w', encoding = 'utf8', newline='') as outcsv:   
        writer = csv.writer(outcsv, delimiter='\t', quotechar = '"')
        for row in user_posts_oversampled:
            writer.writerow(row)

with open("C:\\CLPsych Challenge\\Dataset\\PreProcessing\\Task A-BERT_FINE_TUNING\\Pre-Processed-Data-Normal\\User_Posts_Processed_Test_Final.tsv",'w', encoding = 'utf8', newline='') as outcsv:   
        writer = csv.writer(outcsv, delimiter='\t', quotechar='"')
        for row in final_test_data:
            writer.writerow(row)
