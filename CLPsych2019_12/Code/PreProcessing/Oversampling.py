# -*- coding: utf-8 -*-
"""
Oversampling Class B Samples to Class A
"""
import sys
import csv
import random

csv.field_size_limit(sys.maxsize)
#user_posts = list()
user_posts_final = list()
with open("/home/yy452/rds/rds-gvdd-Yuap0gjVpKM/yy452/CLPsych2019_12/Dataset/task_c/Full_Train_Data.tsv",'r', encoding = 'utf8', newline='') as outcsv:   
        reader = csv.reader(outcsv, delimiter='\t',quotechar = '"')
        for row in reader:
            user_posts_final.append(row)
    
#with open("C:\\CLPsych Challenge\\Dataset\\PreProcessing\\Non-PreProcessed-Data\\User_Posts_Processed_Train_Full_Final.tsv",'r', encoding = 'utf8', newline='') as outcsv:   
#        reader = csv.reader(outcsv, delimiter='\t',quotechar = '"')
#        for row in reader:
#            user_posts.append(row)

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
        
#user_posts_undersampled = over_under_sample(user_posts_final, 'd', 0, 149 - 85)

user_posts_oversampled = over_under_sample(user_posts_final, 'b', 1, 127 - 50)
    

count_label_class = dict()
for row in user_posts_oversampled:
    if(row[2] in count_label_class):
        count_label_class[row[2]] += 1
    else:
        count_label_class[row[2]] = 1

with open("/home/yy452/rds/rds-gvdd-Yuap0gjVpKM/yy452/CLPsych2019_12/Dataset/task_c/Full_Train_Data_Final.tsv",'w', encoding = 'utf8', newline='') as outcsv:   
        writer = csv.writer(outcsv, delimiter='\t', quotechar = '"')
        for row in user_posts_oversampled:
            writer.writerow(row)