AWS_ACCESS_KEY_ID = "AKIAUFQHVDXL6PGMIHTJ" #@param {type:"string"}
AWS_SECRET_ACCESS_KEY = "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" #@param {type:"string"}
S3_UPLOAD_PATH = "s3://my-bucket/sentiment-analyzer/bert" #@param {type:"string"}

import sys
import re

if AWS_ACCESS_KEY_ID == "":
    print("\033[91m{}\033[00m".format("ERROR: Please set AWS_ACCESS_KEY_ID"), file=sys.stderr)

elif AWS_SECRET_ACCESS_KEY == "":
    print("\033[91m{}\033[00m".format("ERROR: Please set AWS_SECRET_ACCESS_KEY"), file=sys.stderr)

else:
    try:
        bucket, key = re.match("s3://(.+?)/(.+)", S3_UPLOAD_PATH).groups()
    except:
        print("\033[91m{}\033[00m".format("ERROR: Invalid s3 path (should be of the form s3://my-bucket/path/to/file)"), file=sys.stderr)

import os
import boto3

s3 = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

filepath = "/home/yy452/rds/rds-gvdd-Yuap0gjVpKM/yy452/CLPsych2019_12/output/task_a_emoji/bert_expert/bert.bin"
filekey = "bert.bin"
print("Uploading s3://{}/{} ...".format(bucket, filekey), end = '')
s3.upload_file(filepath, bucket, filekey)
print(" ✓")

print("\nUploaded model export directory to " + S3_UPLOAD_PATH)