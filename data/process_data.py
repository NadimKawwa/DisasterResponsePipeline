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
import sys


###############
# LOAD DATA   #
###############
def load_data(messages_path='data/disaster_messages.csv', categories_path='data/disaster_categories.csv'):
    """
    Takes in filepaths to two csv files
    assumes filepath in case none provided
    """
    if messages_path is None or categories_path is None:
        sys.exit("At least file path is incorrect!")
        
    #load the data
    messages = pd.read_csv(messages_path)
    
    #drop duplicate ids
    messages_clean = messages.drop_duplicates(subset='id')

    # load categories dataset
    categories = pd.read_csv(categories_path)
    
    #drop duplicates
    categories_clean = categories.drop_duplicates(subset='id')
    
    #assert we have exact 1:1 matching of ids
    assert categories_clean.id.equals(messages_clean.id)
    
    # merge datasets
    df = pd.merge(left= messages_clean, right=categories_clean, how='inner', on='id', validate='one_to_one')

    return df

##############
# CLEAN DATA #
##############

def clean_data (df):
    
    """
    Takes in a merged pandas dataframe and returns a cleaned version
    Input:
    - Pandas dataframe
    Output:
    - Pandas dataframe
    """
    
    # create a dataframe of the 36 individual category columns
    #set expand dimensionality to TRUE
    categories = df.categories.str.split(';', expand=True)
    categories.head()
    
    # select the first row of the categories dataframe
    row = categories.loc[0,:]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.str.split('-').str.get(0)
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        #categories[column] = 

        # convert column from string to numeric
        #categories[column] = 

        #opt to extract numeric portion of entry
        categories[column]=categories[column].str.extract('(\d+)').astype(int)
        
    
    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])
    
    # concatenate the original dataframe with the new `categories` dataframe
    df  = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df = df.drop_duplicates(keep='first')
    
    return df


#############
# SAVE DATA # 
#############



def save_data(df, db_name):
    """
    Saves a pandas dataframe as a sql database
    Input:
    - Pandas dataframe
    - Name for database
    Returns:
    - Database
    """
    
    #instantiate engine
    engine = create_engine('sqlite:///' + db_name)
    
    #create sql table, replace existing table of same name
    df.to_sql('DisasterMessages', con = engine, if_exists='replace')
    
    print("Engine: {}\n Table: {}".format(engine, 'DisasterMessages'))
    

########
# MAIN #
########



def main():
    
    #check for correct input length
    if len(sys.argv) != 4:
        print("Incorrect number of arguments!\n")
        print("Provide: 2 dataset filepaths and one path for database\n")
        sys.exit("Exiting script")
        
    #reads in the arguments 
    messages, categories, db = sys.argv[1:]
    print("reading messages from {} and categories from {}".format(messages, categories))
    df = load_data(messages, categories)
    
    df = clean_data(df)
    
    save_data(df, db)
    

if __name__ == '__main__':
    main()