#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Tweet Text Encoding 

Input: using column "full_text" or column "text" when "full_text" is empty or Nan
Output: csv having columns "tweet_id" and "full_text_encoding"
"""
import pandas as pd
import numpy as np
import math
from sentence_transformers import SentenceTransformer
import re
import sys

# In[2]:


input_file = sys.argv[1]
output_file = sys.argv[2]

# "full_text" column is missing text for few rows, in that case use the "text" column

df_both=pd.read_csv(input_file, usecols = ['tweet_id','text','full_text'], dtype=object)


# In[3]:


for index, row in df_both.iterrows():
    #print("Text:",row['text'])
    #print("full_text:",row['full_text'])
    #print(index)
    if (type(row['full_text'])!=str) | (row['full_text']==" "):
        row['full_text']=row['text']   
    if (type(row['full_text'])!=str) | (row['full_text']==" "):
        df_both.drop(index, inplace = True)
    #row['full_text']=' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",row['full_text']).split())
    #print("_______________Final full_text:",row['full_text']) 


# In[5]:


df_final = df_both[['tweet_id','full_text']]


# In[ ]:


model = SentenceTransformer('sentence-transformers/use-cmlm-multilingual')


# In[ ]:


embeddings = model.encode(df_final['full_text'])


# In[ ]:


df_final["Embeddings"]=''
for i in range(len(embeddings)):
    df_final.at[i, "Embeddings"] = embeddings[i]


# In[ ]:


df_final.to_csv(output_file, index = False)

