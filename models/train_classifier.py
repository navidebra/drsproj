import sys
import re
import pandas as pd
import pickle
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

nltk.download(['punkt', 'stopwords', 'wordnet'])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """
    Loads X and y and category_names from the given database

    Args:
        database_filepath : filepath string of the sqlite database

    Returns:
        X : messages from database in dataframe format
        y : values corresponding to the type of message category in dataframe format
        category_names : category names list for classification
    """

    # creating sqlite engine and reading the specified table
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("messages", engine)

    # define features, label arrays, and category names
    X = df.message.values
    y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = y.columns.tolist()

    return X, y, category_names


def tokenize(text):
    """
    gets raw text string and returns cleaned,  lemmatized, and tokenized text

    Args:
        text : raw string containing the message

    Returns:
        Clean and preprocessed text string
    """

    # initializing necessary text cleaning functions
    stop_words = stopwords.words("english")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # removing punctuations and converting to lower case
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())

    # lemmatizing and removing English stop words
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return clean_tokens


def build_model():
    """Returns the GridSearchCV object to be used as the model

    Args:
        None

    Returns:
        cv: Grid search model object
    """

    # Creating pipeline
    pipeline = Pipeline([

        ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(n_estimators=100, learning_rate=0.6)))

    ])

    # defining parameters to check in gridsearch
    # Uncomment for additional parameters
    parameters = {
        # 'clf__estimator__learning_rate' : (0.6, 1.0),
        # 'clf__estimator__min_samples_split': [2, 4],
        # 'clf__estimator': [AdaBoostClassifier(), RandomForestClassifier()]
        # 'clf__estimator__n_estimators' : [50, 100]
    }

    # Create grid search model object
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Prints multi-output classification results in a dataframe format

    Args:
        model : scikit-learn fitted model
        X_text : X test set
        Y_test : Y test classifications
        category_names : category names

    Returns:
        None
    """

    # Generating predictions
    y_pred = model.predict(X_test)

    y_pred_df = pd.DataFrame(y_pred, columns=category_names)

    # creating results dataframe with appropriate columns
    results_df = pd.DataFrame(columns=['Category', 'Percision', 'Recall', 'F1-score', 'Support'])

    # iterating through rows of classification report
    # putting it all in a dataframe
    for i in range(len(category_names)):
        string = classification_report(Y_test.iloc[:, i], y_pred_df.iloc[:, i])
        tot_loc = string.split().index('total')
        string = string.split()[tot_loc + 1:]
        results_df.loc[len(results_df)] = [category_names[i]] + string

    print(results_df)


def save_model(model, model_filepath):
    """saves the model in the given filepath

    Args:
        model : The fitted scikit-learn model
        model_filepath : string of the filepath to save the model

    Returns:
        None
    """

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()