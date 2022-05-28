import pandas as pd
import csv
from datetime import datetime
from nltk.corpus import stopwords
import re
import unidecode
import json
import os
import argparse
from collections import defaultdict
import nltk
import sys
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from nltk import ngrams


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

prp_dict = {1: ['i feel', "i want", 'i wanted', 'i can', 'i could', 'i cant', 'myself', 'my life'],
       3: ['they felt','they feels', 'they wanted', 'they wants','kill themself', 'i miss', 'he felt', 'he feels','he has','he wants', 'he wanted','she wants','she has', 'she felt', 'she feels', 'she wanted', 'herself', 'himself','my friend', 'best friend']}


def ws_tokenizer(text):
    return text.split()


def count_prps(post):
    texts = nltk.word_tokenize(post)
    
    prps = dict()
    for word in texts:
        word = word.lower()
        if word in prp_dict[1]:
            if word in prps.keys():
                prps[word] += 1
            else:
                prps[word] = 1
        elif word in prp_dict[3]:
            if word in prps.keys():
                prps[word] += 1
            else:
                prps[word] = 1
                
    n = 2
    two_grams = ngrams(post.split(), n)
    words = []
    for grams in two_grams:
        words.append(grams[0] + ' ' + grams[1])
        
    for word in words:
        word = word.lower()
        if word in prp_dict[1]:
            if word in prps.keys():
                prps[word] += 1
            else:
                prps[word] = 1
        elif word in prp_dict[3]:
            if word in prps.keys():
                prps[word] += 1
            else:
                prps[word] = 1
       
    return prps

def lower_dict(d):
    new_dict = {}
    for k, v in d.items():
        if k.lower() in new_dict:
            new_dict[k.lower()] += v
        else:
            new_dict[k.lower()] = v
    return new_dict

def count_third(prps):
    c1=0
    c3 = 0
    for word, count in prps.items():
        if word in prp_dict[1]:
            c1 += count
        elif word in prp_dict[3]:
            c3 += count
    return c3 > c1

parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--raw_input", default="",required=True, type=str, help="The original input text path. format: txt")	
parser.add_argument("--output_dir", default="",required=True, type=str, help="The original input text path. format: txt")	

args = parser.parse_args()
with open(args.raw_input, 'r') as file:
    post = file.readlines()[0]
output_dir = args.output_dir

post = " ".join(text_processor.pre_process_doc(str(post)))
prps = count_prps(post)

if count_third(prps):
    output_file = os.path.join(output_dir, "result.csv")
    outcsv = open(output_file,'w', encoding = 'utf8', newline='')
    writer = csv.writer(outcsv,quotechar = '"')
    writer.writerow(["User","results"])
    writer.writerow([1,0])
    outcsv.close()
    
    sys.exit(1)