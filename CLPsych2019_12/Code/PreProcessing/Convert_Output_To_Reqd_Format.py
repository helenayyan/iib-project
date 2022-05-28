"""
Convert Output from BERT to Labels in Challenge
"""

import csv

labels = list()
user_id = list()
with open("C:\\CLPsych Challenge\\Dataset\\PreProcessing\\Task C-BERT_FINE_TUNING\\Test_Reqd_Labels.csv",'r', encoding = 'utf8', newline='') as outcsv:   
        reader = csv.reader(outcsv, delimiter=',',quotechar = '"')
        for i, row in enumerate(reader):
            if(i == 0):
                continue
            labels.append(row[2])
            user_id.append(row[0])

mapper = {
        '0':'a',
        '4':'a',
        '1':'b',
        '2':'c',
        '3':'d'}
            
with open("C:\\CLPsych Challenge\\Dataset\\PreProcessing\\Task C-BERT_FINE_TUNING\\ASU.BERT_FINE_TUNING.task_C.out.csv",'w', encoding = 'utf8', newline='') as outcsv:   
        writer = csv.writer(outcsv, delimiter=',', quotechar = '"')
        for user_id, label in zip(user_id, labels):
            writer.writerow([user_id, mapper[label]])