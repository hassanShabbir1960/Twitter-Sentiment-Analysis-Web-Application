#!/usr/bin/env python
# coding: utf-8

# In[170]:


## For any task data collection is considered very vital part. The first phase of our project was to perform sentimental analysis
## on tweets. So for that, we looked for the open source labelled tweets. We collected the data from various open sources. Data
## downloaded from different sources has different class distribution ie (Different number of tweets for all classes) . Since non
## uniform data can make our model biased towards one class, so we take equal amount of tweets for each class , to have a unbiased
## model that can perform the sentimental analysis.

## Here are the links from which we have gathered our data:

## https://www.kaggle.com/kazanova/sentiment140
## https://www.kaggle.com/shashank1558/preprocessed-twitter-tweets
## https://www.kaggle.com/crowdflower/first-gop-debate-twitter-sentiment


# In[152]:


import pandas as pd


## Reading the file
df1= pd.read_csv("Sentimental_analysisdata.csv",encoding='ISO-8859-1',header=None)
## Renaming the column names of file , so that our final file has same column names
df1.rename(columns={0: 'sentiment', 5: 'text'}, inplace=True)
## Renaming the label values, so that we have same label values for all files
df1['sentiment'] = df1['sentiment'].map({0: 'negative', 4: "positive"})
## Fetching some of the tweets from the file
a= df1[df1['sentiment']=='negative'][:2851]
b= df1[df1['sentiment']=='positive'][:2851]
## Fetching only two columns from the data for both classes
file1= pd.concat( [a.filter(['sentiment', 'text']),b.filter(['sentiment', 'text']) ] ,axis=0)
## Saving the file as csv
file1.to_csv('file1.csv',index=None)


# In[161]:


df2=pd.read_csv('Sentiment.csv') ## Reading the file
## Renaming the label values, so that we have same label values for all files
df2['sentiment'] = df2['sentiment'].map({'Negative': 'negative', 'Positive': "positive",'Neutral':'neutral'})

## Fetching some of the tweets from the file
a= df2[df2['sentiment']=='negative'][:2236]
b= df2[df2['sentiment']=='positive'][:2236]
c= df2[df2['sentiment']=='neutral'][:2990]
## Fetching only two columns from the data for both classes
file2= pd.concat( [a.filter(['sentiment', 'text']),b.filter(['sentiment', 'text']) ,c.filter(['sentiment', 'text']) ] ,axis=0)
## Saving the file as csv
file2.to_csv('file2.csv',index=None)


# In[162]:


df3=pd.read_csv('Tweets.csv') ## Reading the file
## Renaming the column names of file , so that our final file has same column names
df3.rename(columns={'airline_sentiment': 'sentiment'}, inplace=True)

## Fetching some of the tweets from the file
a= df3[df3['sentiment']=='negative'][:2363]
b= df3[df3['sentiment']=='positive'][:2363]
c= df3[df3['sentiment']=='neutral'][:2990]

## Fetching only two columns from the data for both classes
file3= pd.concat( [a.filter(['sentiment', 'text']),b.filter(['sentiment', 'text']) ,c.filter(['sentiment', 'text']) ] ,axis=0)
## Saving the file as csv
file3.to_csv('file3.csv',index=None)


# In[109]:


a= (pd.read_csv("processedNeutral.csv")) ## Reading the file
a= list(a.columns) ## In this file values are stored as columns, so storing all values in an array

import numpy as np
lab= np.zeros(len(a)) ## Assigning labels

df= pd.DataFrame(lab,columns=['sentiment'])
df['text']= a
df['sentiment'] = df['sentiment'].map({0: 'neutral'}) ## Setting the same name for labels
## Saving the dataframe as csv
df.to_csv("file4.csv",index=None)


# In[167]:


## Merging all the files to make one file for whole data
df1=pd.read_csv('file1.csv')
df2=pd.read_csv('file2.csv')
df3=pd.read_csv('file3.csv')
df4=pd.read_csv('file4.csv')

df= pd.concat([df1,df2,df3,df4],axis=0)
df.to_csv("finalfile.csv",index=None)

