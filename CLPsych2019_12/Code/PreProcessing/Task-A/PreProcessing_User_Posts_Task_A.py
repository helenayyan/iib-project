"""
Expands contractions, removes stopwords, etc and writes the posts in BERT required Input format
Input Files:
word_contractions_expansion.json, combined_data_Task_A.csv, trainUserIds_TaskA_Final.csv OR testUserIds_TaskA_Final.csv, task_A_train.posts.csv
Output Files:
User_Posts_Processed_Train.csv
"""

import csv
from datetime import datetime
from nltk.corpus import stopwords
import nltk
import re
import unidecode
import json
#import random
from collections import defaultdict
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

vader = SentimentIntensityAnalyzer()

with open('/home/yy452/rds/rds-gvdd-Yuap0gjVpKM/yy452/CLPsych2019_12/Code/PreProcessing/contractions.json') as f:
    cList = json.load(f)

c_re = re.compile('(%s)' % '|'.join(cList.keys()))

sw = stopwords.words("english")
extra_stop_words = ["cannot", "could", "would", "us", "may", "might", "need", "ought", "shall", "alls", "n't", "'s", "'ve", "'t", "'m", "'d", "'ll", "t"]
sw.extend(extra_stop_words)
#sw = []

def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)

def humanize_unixtime(unix_time):
    time = datetime.fromtimestamp(int(unix_time)).strftime('%d-%m-%Y %H.%M')
    return time

def word_cleaner(word):
    word = unidecode.unidecode(word)
    if(word.lower() in sw):
        word = " "
        
    word = word.replace("_PERSON_", " ")
    word = word.replace("_IP_", " ")
    word = word.replace("_EMAIL_", " ")
    word = word.replace("_URL_", " ")
    word = word.replace(".", " ")
    p = re.compile('([A-Za-z]+)[.]')
    word = p.sub(r'\1 ', word)
    p = re.compile('[.]([A-Za-z]+)')
    word = p.sub(r' \1', word)
    word = word.replace("!", " ")
    word = word.replace(",", " ")
    word = word.replace("/", " ")
    word = word.replace("~", " ")
    # word = word.replace("-", " ")
    word = word.replace("--", " ")
    word = word.replace("(", " ")
    word = word.replace(")", " ")
    word = word.replace("#", " ")
    word = word.replace("?", " ")
    word = word.replace("..", " ")
    word = word.replace("...", " ")
    word = word.replace("’", " ")
    word = word.replace(":", " ")
    word = word.replace("[", " ")
    word = word.replace("]", " ")
    word = word.replace("*", " ")
    word = word.replace("\"", " ")
    word = word.replace("&", " ")
    word = word.replace("{", " ")
    word = word.replace("}", " ")
    word = word.replace("@", " ")
    word = word.replace("↑", " ")
    word = word.replace("$", " ")
    word = word.replace("^", " ")
    word = word.replace("\n", " ")
    word = word.replace("\t", " ")
    word = word.replace("\r", " ")
    word = word.replace("`", " ")
    word = word.replace("'", " ")
    word = word.replace(";", " ")
    if(word == "." or word == " ." or word == " . " or word == ". "):
        word = " "
    return word

path ="/home/yy452/rds/rds-gvdd-Yuap0gjVpKM/yy452/CLPsych2019_12/Dataset/add_expert/sw/combined_data_sw.csv"
all_data = dict()
file = open(path, 'r', encoding = 'utf8')
reader_data = csv.reader(file)
for i, row in enumerate(reader_data):
    if(i == 0):
        continue
    all_data[(row[0], row[1])] = row


train_user_label_path ="/home/yy452/rds/rds-gvdd-Yuap0gjVpKM/yy452/CLPsych2019_12/Dataset/add_expert/sw/train_user_labels.csv"
file =open(train_user_label_path, 'r', encoding = 'utf8')
reader_train = csv.reader(file, delimiter=',')
train_user_id_label = dict()
for row in reader_train:
    train_user_id_label[row[0]] = row[1]


test_user_label_path ="/home/yy452/rds/rds-gvdd-Yuap0gjVpKM/yy452/CLPsych2019_12/Dataset/task_a/testUserIds_TaskA_Final.csv"
file =open(test_user_label_path, 'r', encoding = 'utf8')
reader_test = csv.reader(file, delimiter=',')
test_user_id_label = dict()
for row in reader_test:
    test_user_id_label[row[0]] = row[1]


taskA_path ="/home/yy452/rds/rds-gvdd-Yuap0gjVpKM/yy452/umd_reddit_suicidewatch_dataset_v2/crowd/train/task_A_train.posts.csv"

all_train_posts_of_users_combined = list()
all_train_posts_of_users_combined.append(["User ID", "Post", "Label"])

all_test_posts_of_users_combined = list()
all_test_posts_of_users_combined.append(["User ID", "Post", "Label"])


file =open(taskA_path, 'r', encoding = 'utf8')
reader_user = csv.reader(file, delimiter=',')
taskA_user_posts = defaultdict(list)
for i, row in enumerate(reader_user):
    if(i == 0):
        continue
    taskA_user_posts[row[1]].append(row[0])

just_user_posts_train = list()
just_user_posts_test = list()
user_posts_for_fine_tuning_mlm = list()

count_len_before_g384 = dict()
count_len_before_l384 = dict()
count_len_after_g384 = dict()
count_len_after_l384 = dict()

for user in taskA_user_posts:
    user_posts = list()
    for row in taskA_user_posts[user]:
        user_posts.append(all_data[(row, user)])
    posts_sorted_by_date = sorted(user_posts, key=lambda x : x[3], reverse=True)
    user_post_combined = ""
    for i, post in enumerate(posts_sorted_by_date):
        user_id = post[1]
        length = len(nltk.word_tokenize(post[5])) + len(nltk.word_tokenize(post[4]))
        if length > 384 :
            if length in count_len_before_g384:
                count_len_before_g384[length] += 1
            else:
                count_len_before_g384[length] = 1
        else:
            if length in count_len_before_l384:
                count_len_before_l384[length] += 1
            else:
                count_len_before_l384[length] = 1
            
        post[4] = expandContractions(post[4])
        post[4] =' '.join(post[4].split('\t'))
        post[4] ='.'.join(post[4].split('\n'))
        post[4] =' '.join(post[4].split('|'))
        post[4] =' '.join(post[4].split('\r'))
        
        post[5] = expandContractions(post[5])
        post[5] =' '.join(post[5].split('\t'))
        post[5] ='.'.join(post[5].split('\n'))
        post[5] =' '.join(post[5].split('|'))
        post[5] =' '.join(post[5].split('\r'))
        
        word_tokenized_title = nltk.word_tokenize(post[4])
        word_tokenized_post = nltk.word_tokenize(post[5])
        one_post = ""
        for word in word_tokenized_title:
            user_post_combined += word_cleaner(word) + " "
            one_post += word_cleaner(word) + " "
        
        for word in word_tokenized_post:
            user_post_combined += word_cleaner(word) + " "
            one_post += word_cleaner(word) + " "
            
        length = len(nltk.word_tokenize(one_post))
        if length > 384 :
            if length in count_len_after_g384:
                count_len_after_g384[length] += 1
            else:
                count_len_after_g384[length] = 1
        else:
            if length in count_len_after_l384:
                count_len_after_l384[length] += 1
            else:
                count_len_after_l384[length] = 1
            
    user_post_combined = re.sub(' +', ' ', user_post_combined)
    #user_post_combined = ' '.join(user_post_combined.split(' '))
    user_post_combined = user_post_combined.strip()
    user_post_combined = user_post_combined.lower()
    for sent in nltk.sent_tokenize(user_post_combined):
        user_posts_for_fine_tuning_mlm.append(sent)
    user_posts_for_fine_tuning_mlm.append(None)
    #print(user_post_combined)
    #print("\n\n\n")
    #label = random.randint(0,1)
    if user in train_user_id_label:
        label = train_user_id_label[user]
        all_train_posts_of_users_combined.append([user_id, user_post_combined, label])
        just_user_posts_train.append(user_post_combined)
    else:
        label = test_user_id_label[user]
        all_test_posts_of_users_combined.append([user_id, user_post_combined, label])
        just_user_posts_test.append(user_post_combined)

with open("/home/yy452/rds/rds-gvdd-Yuap0gjVpKM/yy452/CLPsych2019_12/Dataset/add_expert/sw/User_Posts_Processed_Train_Full_Final.tsv",'w', encoding = 'utf8', newline='') as outcsv:   
       writer = csv.writer(outcsv, delimiter='\t',quotechar = '"')
       for row in all_train_posts_of_users_combined:
           writer.writerow(row)

#with open("C:\\CLPsych Challenge\\Dataset\\PreProcessing\\Non-PreProcessed-Data\\User_Posts_Processed_Test_Final.tsv",'w', encoding = 'utf8', newline='') as outcsv:
#        writer = csv.writer(outcsv, delimiter='\t', quotechar = '"')
#        for row in all_test_posts_of_users_combined:
#            writer.writerow(row)

#with open("C:\\CLPsych Challenge\\Dataset\\PreProcessing\\Full_Train_Data.tsv",'w', encoding = 'utf8', newline='') as outcsv:   
#        writer = csv.writer(outcsv, delimiter='\t', quotechar = '"')
#        for row in all_test_posts_of_users_combined:
#            writer.writerow(row)
#        for i, row in enumerate(all_train_posts_of_users_combined):
#            if(i == 0):
#                continue
#            writer.writerow(row)
       
#with open("C:\\CLPsych Challenge\\Dataset\\PreProcessing\\Only_User_Posts_Train_Sentiment.txt",'w', encoding = 'utf8', newline='') as outcsv:   
#        writer = csv.writer(outcsv,quotechar = '"')
#        for row in just_user_posts_train:
#            writer.writerow([row])

# with open("/home/yy452/rds/rds-gvdd-Yuap0gjVpKM/yy452/CLPsych2019_12/Dataset/task_a/Only_User_Posts_For_Fine_Tuning_Without_Sent_Tokenized.txt",'w', encoding = 'utf8', newline='') as outcsv:
#         writer = csv.writer(outcsv, quotechar = '"')
#         for row in user_posts_for_fine_tuning_mlm:
#             if(row == None):
#                 writer.writerow("")
#             else:
#                 writer.writerow([row])

len(all_test_posts_of_users_combined) + len(all_train_posts_of_users_combined)

count_label_class = dict()
for row in all_train_posts_of_users_combined:
    if(row[2] in count_label_class):
        count_label_class[row[2]] += 1
    else:
        count_label_class[row[2]] = 1
        
for row in all_test_posts_of_users_combined:
    if(row[2] in count_label_class):
        count_label_class[row[2]] += 1
    else:
        count_label_class[row[2]] = 1