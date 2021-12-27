#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import dependencies
import pandas as pd
import numpy as np
import os
import csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
analyzer = SentimentIntensityAnalyzer()


# In[4]:


#read csv files into dataframes
csv_path = "https://raw.githubusercontent.com/subhasushi/Company_Reviews_Sentiment_Analysis/master/google-amazon-facebook-employee-reviews/employee_reviews.csv"
company_reviews = pd.read_csv(csv_path)
clean_df = company_reviews.drop("Unnamed: 0", axis = 1)

#delete the unnamed column and set index
clean_df.reset_index()

#decided to remove the rows with missing columns
company_reviews_clean = clean_df.dropna(how="any")
company_reviews_clean.head()


# In[5]:


# filter the data to just display one company at a time
google_reviews = company_reviews_clean.loc[company_reviews_clean["company"]=='google',:]
google_reviews.head()


# In[6]:


company_list = company_reviews_clean["company"].unique()
company_list


# In[7]:


#variables for holding each sentiment for each sentence
google_compound_list = []
google_positive_list = []
google_negative_list = []
google_neutral_list = []

for sentence in google_reviews["summary"]: 
    
     
#     caluculating the polarity scores on each sentence
    results = analyzer.polarity_scores(sentence)
    compound = results["compound"]
    pos = results['pos']
    neg = results['neg']
    neu = results['neu']

    #add each value to the appropriate array
    google_compound_list.append(compound)
    google_positive_list.append(pos)
    google_negative_list.append(neg)
    google_neutral_list.append(neu)

#store the average sentiments
sentiments = pd.DataFrame([{"Company":"Google",
             "Compound": np.mean(google_compound_list),
             "Positive":np.mean(google_positive_list),
             "Negative": np.mean(google_negative_list),
             "Neutral": np.mean(google_neutral_list),
             "Review Count": len(google_compound_list)}])
sentiments = sentiments.set_index("Company")

print(sentiments)


# In[21]:


#Amazon reviews analyzer
amazon_reviews = company_reviews_clean.loc[company_reviews_clean["company"]=="amazon",:]

#empty list to hold each sentiment
amazon_compound_list = []
amazon_positive_list = []
amazon_negative_list = []
amazon_neutral_list = []

#loop through the summary column and pass it into the analyser
for sentence in amazon_reviews["summary"]:
    results = analyzer.polarity_scores(sentence)
    compound = results["compound"]
    pos = results["pos"]
    neg = results["neg"]
    neu = results["neu"]
    
    #add each of these into the individual list
    amazon_compound_list.append(compound)
    amazon_positive_list.append(pos)
    amazon_negative_list.append(neg)
    amazon_neutral_list.append(neu)
    
#store average sentiments
sentiments = {"Company":"Amazon",
              "Compound":np.mean(amazon_compound_list),
              "Positive":np.mean(amazon_positive_list),
              "Negative":np.mean(amazon_negative_list),
              "Neutral":np.mean(amazon_neutral_list),
              "Number of Reviews":len(amazon_compound_list)}

print(sentiments)   

#sentiments.loc["Amazon"] = [np.mean(amazon_compound_list),np.mean(amazon_positive_list),np.mean(amazon_negative_list),np.mean(amazon_neutral_list),len(amazon_compound_list)]


# In[9]:


#analyzer for facebook
fb_compound_list = []
fb_positive_list = []
fb_negative_list = []
fb_neutral_list = []
facebook_reviews = company_reviews_clean.loc[company_reviews_clean["company"]=="facebook"]
for each in facebook_reviews["summary"]:
    result = analyzer.polarity_scores(each)
    compound = result["compound"]
    pos = result["pos"]
    neg = result["neg"]
    neu = result["neu"]
    
    #store each sentiment in respective lists
    fb_compound_list.append(compound)
    fb_positive_list.append(pos)
    fb_negative_list.append(neg)
    fb_neutral_list.append(neu)
    
#average the sentiments
# sentiments = {"Company":"Facebook",
#              "Compound":np.mean(fb_compound_list),
#              "Positive":np.mean(fb_positive_list),
#              "Negative":np.mean(fb_negative_list),
#              "Neutral":np.mean(fb_neutral_list),
#              "Number of Reviews":len(fb_compound_list)}
sentiments.loc["Facebook"] = [np.mean(fb_compound_list),np.mean(fb_positive_list),np.mean(fb_negative_list),np.mean(fb_neutral_list),len(fb_compound_list)]
    


# In[12]:


#Netflix reviews analyzer
nf_compound_list = []
nf_positive_list = []
nf_negative_list = []
nf_neutral_list = []
nf_reviews = company_reviews_clean.loc[company_reviews_clean["company"]=="netflix"]


#loop through each sentence
for each in nf_reviews["summary"]:
    result = analyzer.polarity_scores(each)
    comp = result["compound"]
    pos = result["pos"]
    neg = result["neg"]
    neu = result["neu"]
    
    #append it to theie respective sentiment list
    nf_compound_list.append(comp)
    nf_positive_list.append(pos)
    nf_negative_list.append(neg)
    nf_neutral_list.append(neu)
    
#average the sentiments to get overall scores
# sentiments = {"Company":"Netflix",
#              "Compound":np.mean(nf_compound_list),
#              "Positive":np.mean(nf_positive_list),
#              "Negative":np.mean(nf_negative_list),
#              "Neutral":np.mean(nf_neutral_list),
#              "Number of Reviews":len(nf_compound_list)}
sentiments.loc["Netflix"] = [np.mean(nf_compound_list),np.mean(nf_positive_list),np.mean(nf_negative_list),np.mean(nf_neutral_list),len(nf_compound_list)]


# In[13]:


#Apple reviews analyzer
apple_compound_list = []
apple_positive_list = []
apple_negative_list = []
apple_neutral_list = []
apple_reviews = company_reviews_clean.loc[company_reviews_clean["company"]=="apple"]

# loop through each sentence
for each in apple_reviews["summary"]:
    result = analyzer.polarity_scores(each)
    comp = result["compound"]
    pos = result["pos"]
    neg = result["neg"]
    neu = result["neu"]
    
    #append it to theie respective sentiment list
    apple_compound_list.append(comp)
    apple_positive_list.append(pos)
    apple_negative_list.append(neg)
    apple_neutral_list.append(neu)
    
#average the sentiments to get overall scores
# sentiments = {"Company":"Apple",
#              "Compound":np.mean(apple_compound_list),
#              "Positive":np.mean(apple_positive_list),
#              "Negative":np.mean(apple_negative_list),
#              "Neutral":np.mean(apple_neutral_list),
#              "Number of Reviews":len(apple_compound_list)}
sentiments.loc["Apple"] = [np.mean(apple_compound_list),np.mean(apple_positive_list),np.mean(apple_negative_list),np.mean(apple_neutral_list),len(apple_compound_list)]


# In[14]:


#Microsoft reviews analyzer
ms_compound_list = []
ms_positive_list = []
ms_negative_list = []
ms_neutral_list = []
ms_reviews = company_reviews_clean.loc[company_reviews_clean["company"]=="microsoft"]

# loop through each sentence
for each in ms_reviews["summary"]:
    result = analyzer.polarity_scores(each)
    comp = result["compound"]
    pos = result["pos"]
    neg = result["neg"]
    neu = result["neu"]
    
    #append it to theie respective sentiment list
    ms_compound_list.append(comp)
    ms_positive_list.append(pos)
    ms_negative_list.append(neg)
    ms_neutral_list.append(neu)
    
#average the sentiments to get overall scores
# sentiments = {"Company":"Microsoft",
#              "Compound":np.mean(ms_compound_list),
#              "Positive":np.mean(ms_positive_list),
#              "Negative":np.mean(ms_negative_list),
#              "Neutral":np.mean(ms_neutral_list),
#              "Number of Reviews":len(ms_compound_list)}
sentiments.loc["Microsoft"] = [np.mean(ms_compound_list),np.mean(ms_positive_list),np.mean(ms_negative_list),np.mean(ms_neutral_list),len(ms_compound_list)]
print(sentiments)


# In[17]:


#arrange as per greater number of positive ratings
sentiments.sort_values(["Positive"],
                      axis = 0,
                      ascending=[False],
                      inplace=True)
print(sentiments)


# In[ ]:




