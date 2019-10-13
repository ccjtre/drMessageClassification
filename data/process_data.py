import sys
import pandas as pd
from sqlalchemy import create_engine

# TODO: db write verification, remove resp=2

def load_data(messages_filepath, categories_filepath):
    """ Returns combined messages/categories DataFrame from input files
    
    Parameters:
        messages_filepath (str): path to messages csv.
        categories_filepath (str): path to categories csv.
   
   Returns:
    	categorized_messages(DataFrame):messages joined with categories, based on 'id'   

    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    categorized_messages = messages.join(categories.set_index('id'), on='id')
    return categorized_messages


def clean_data(df):
    """ Expands categories column and remove dups from load_data output (df)"""
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand=True)
    # use elements of first row of categories for renaming the expanded columns
    row = categories.iloc[0]
    category_colnames = list(row.apply(lambda x: x.split('-')[0]))
    categories.columns = category_colnames
    # set each value of category columns be the last character of the string
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop the original categories column from df
    df.drop(columns=['categories'], inplace=True)
    # concatenate df with the expanded categories
    df = pd.concat([df, categories], axis=1)
    # drop where related=2
    df = df[df['related'] != 2]
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """ Writes df to database_filename.
    Parameters:
        df (DataFrame): output of clean_data.
        database_filename (str): path to database file.
    Returns:
        None

    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('categorized_messages', engine, index=False)
    print("table saved, name: categorized_messages")

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()