# -*- coding: utf-8 -*-
"""
Do an SQL join on the the dataset by joining on 'user_id', 'post_id' and 'subreddit'
"""

import pandas as pd

df1 = pd.read_csv("/home/yy452/rds/rds-gvdd-Yuap0gjVpKM/yy452/umd_reddit_suicidewatch_dataset_v2/crowd/test/task_C_test.posts.csv")
df2 = pd.read_csv("/home/yy452/rds/rds-gvdd-Yuap0gjVpKM/yy452/umd_reddit_suicidewatch_dataset_v2/crowd/test/shared_task_posts_test.csv")

merged_data = pd.merge(df1, df2, on=['post_id'])
merged_data = merged_data.drop(merged_data.columns[[1,2]], axis=1) 
merged_data.to_csv("/home/yy452/rds/rds-gvdd-Yuap0gjVpKM/yy452/CLPsych2019_12/Dataset/task_c/combined_data_Task_C_Test.csv", index=False, encoding='utf-8')

# merged_data['user_id'].nunique()

# df1 = pd.read_csv("/home/yy452/rds/rds-gvdd-Yuap0gjVpKM/yy452/umd_reddit_suicidewatch_dataset_v2/crowd/train/task_C_train.posts.csv")
# df2 = pd.read_csv("/home/yy452/rds/rds-gvdd-Yuap0gjVpKM/yy452/umd_reddit_suicidewatch_dataset_v2/crowd/train/shared_task_posts.csv")

# merged_data = pd.merge(df1, df2, on=['post_id', 'user_id','subreddit'])

# merged_data.to_csv("/home/yy452/rds/rds-gvdd-Yuap0gjVpKM/yy452/CLPsych2019_12/Dataset/task_c/combined_data_Task_C.csv", index=False, encoding='utf-8')
