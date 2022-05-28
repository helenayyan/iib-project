import pandas as pd
import csv
from datetime import datetime
from nltk.corpus import stopwords
import re
import unidecode
import json
import os
import argparse
#import random
from collections import defaultdict
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from nltk import ngrams

def ws_tokenizer(text):
    return text.split()


text_processor = TextPreProcessor(
    normalize=[],
    annotate={},
    all_caps_tag="wrap",
    fix_text=True,
    segmenter="twitter_2018",
    corrector="twitter_2018",
    unpack_hashtags=True,
    unpack_contractions=True,
    spell_correct_elong=True,
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    # tokenizer=ws_tokenizer,
    dicts=[emoticons]
)

parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--raw_input", default="",required=True, type=str, help="The original input text path. format: txt")	
parser.add_argument("--output_dir", default="",required=True, type=str, help="The original input text path. format: txt")	

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
    word = word.replace("_person_", " ")
    word = word.replace("_ip_", " ")
    word = word.replace("_email_", " ")
    word = word.replace("_number_", " ")
    word = word.replace("_percent_", " ")
    word = word.replace("_money_", " ")
    word = word.replace("_time_", " ")
    word = word.replace("_date_", " ")
    word = word.replace("_url_", " ")
    word = word.replace(".", " ")
    p = re.compile('([A-Za-z]+)[.]')
    word = p.sub(r'\1 ', word)
    p = re.compile('[.]([A-Za-z]+)')
    word = p.sub(r' \1', word)
    word = word.replace("!", " ")
    word = word.replace(" r ", " ")
    word = word.replace(",", " ")
    word = word.replace("/", " ")
    word = word.replace("~", " ")
    word = word.replace("-", " ")
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

args = parser.parse_args()
with open(args.raw_input, 'r') as file:
    post = file.readlines()[0]


post = " ".join(text_processor.pre_process_doc(post))
post = expandContractions(post)
post =' '.join(post.split('\t'))
post ='.'.join(post.split('\n'))
post =' '.join(post.split('|'))
post =' '.join(post.split('\r'))

word_tokenized_post = nltk.word_tokenize(post)
user_post_combined = ""
for word in word_tokenized_post:
    user_post_combined += word_cleaner(word) + " "
print(user_post_combined)
user_post_combined = re.sub(' +', ' ', user_post_combined)
user_post_combined = user_post_combined.strip()
user_post_combined = user_post_combined.lower()
print(user_post_combined)
all_test_posts_of_users_combined = list()
all_test_posts_of_users_combined.append(["User ID", "Post"])
all_test_posts_of_users_combined.append([1, user_post_combined])

# with open(args.output_dir + "/User_Posts_Processed.txt",'w') as f:   
#     f.write(user_post_combined)
with open(args.output_dir + "/Full_Test_Data.tsv",'w') as outcsv:           
    writer = csv.writer(outcsv, delimiter='\t', quotechar = '"')
    for row in all_test_posts_of_users_combined:
        writer.writerow(row)
