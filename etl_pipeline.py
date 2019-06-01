#Imports disaster messages
#cleans up the data
#exportst the file as a sql database

###########
# IMPORT  #
###########

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import datetime

# load messages dataset
messages = pd.read_csv('messages.csv')

# load categories dataset
categories = pd.read_csv('categories.csv')

###########
# MERGE  #
###########

#get unique ids for dataframes
messages_clean = messages.drop_duplicates(subset='id')
categories_clean = categories.drop_duplicates(subset='id')

#assert we have exact 1:1 matching of ids
assert categories_clean.id.equals(messages_clean.id)

# merge datasets
df = pd.merge(left= messages_clean, right=categories_clean, how='inner', on='id', validate='one_to_one')

#####################
# SPLIT CATEGORIES  #
#####################

# create a dataframe of the 36 individual category columns
#set expand dimensionality to TRUE
categories = df.categories.str.split(';', expand=True)

# select the first row of the categories dataframe
row = categories.loc[0,:]

# use this row to extract a list of new column names for categories.
# up to the second to last character of each string with slicing
category_colnames = row.str.split('-').str.get(0)

# rename the columns of `categories`
categories.columns = category_colnames

##############################
# CONVERT CATEGORIES to 1/0  #
##############################

for column in categories:
    #opt to extract numeric portion of entry
    categories[column]=categories[column].str.extract('(\d+)').astype(int)
    
#######################
# REPLACE CATEGORIES  #
#######################

# drop the original categories column from `df`
df = df.drop(columns=['categories'])

# concatenate the original dataframe with the new `categories` dataframe
df  = pd.concat([df, categories], axis=1)

# drop duplicates
df = df.drop_duplicates(keep='first')

#########################
# SAVE TO SQL DATABASE  #
#########################

#instantiate engine
engine = create_engine('sqlite:///messages.db')
#create sql table, replace existing table of same name
df.to_sql('messages', engine, index=False, if_exists='replace')


#print end of script
print("Success! messages.db created on {}".format(datetime.datetime.now()))