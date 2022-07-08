import sys
import pandas as pd
import sqlalchemy


def load_data(messages_filepath, categories_filepath):
    # using pandas load messages data to dataframe
    df_message = pd.read_csv(messages_filepath)
    # using pandas load categories data to dataframe
    df_category = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(left=df_message, right=df_category, on="id")
    return df


def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)
    # create categories df column name
    categories.columns = categories.iloc[0].apply(lambda x: x[:-2]).values
    # Convert category values to just numbers 0 or 1.
    categories = categories.applymap(lambda x: int(x[-1]))
    # category values [0,1,2] to just numbers 0 or 1
    categories["related"] = categories["related"].apply(lambda x: 0 if x == 0 else 1)
    # drop the original categories column df
    df.drop(columns=["categories"], inplace=True)
    # concatenate dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    engine = sqlalchemy.create_engine('sqlite:///' + str(database_filename))
    df.to_sql("disaster", engine, index=False, if_exists='replace')
    return True


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
