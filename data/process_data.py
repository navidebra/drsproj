import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    loads the labeled data from different sources and merges them together in a dataframe
    
    Args:
        messages_filepath : string of the file path of messages csv file
        categories_filepath : string of the file path of categories csv file
        
    Returns:
   
        df : merged messages and categories dataframe on id column
    """

    # Loading the appropriate files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merging two files on id
    df = pd.merge(categories, messages, on='id')

    return df


def clean_data(df):
    """Cleans data:
        - cleans and separates categories into individual columns
        - converts categories to binary values
        - drop duplicates
    
    Args:
        df : categories and messages combined dataframe
        
    Returns:
        df : cleaned dataframe with separated categories
    """

    # Cleaning Process
    # Splitting values and expanding categories columns
    categories = df['categories'].str.split(pat=';', expand=True)

    # Extracting category names
    row = categories.iloc[0].tolist()

    # removing the last two characters in each row
    category_colnames = [(lambda x: x[:-2])(x) for x in row]

    # updating column names to category names
    categories.columns = category_colnames

    # iterating through each column
    for column in categories:
        # setting each value to be the last character of the string
        categories[column] = [(lambda x: x[-1:])(x) for x in categories[column]]

        # converting column from string to numeric
        categories[column] = pd.to_numeric(categories[column])


    categories.related.replace(2, 1, inplace=True)

    # dropping old categories column
    df.drop('categories', axis=1, inplace=True)

    # concatenating new categories to data
    df = pd.concat([df, categories], axis=1)

    # dropping duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """saves the preprocessed data to a sqlite database
    
    Args:
        df : cleaned dataframe
        database_filename : string of the file path to save the database
        
    Returns:
        None
    """

    # Creating sqlite engine
    engine = create_engine(f'sqlite:///{database_filename}')

    # saving passed dataframe to the specified table in database
    df.to_sql('messages', engine, index=False, if_exists='replace')


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
