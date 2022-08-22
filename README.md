# Disaster Response Message Classification Pipelines

### Table of Contents

1. [Libraries](#Libraries)
2. [Project Description](#ProjectDescription)
3. [File Description](#FileDescription)
4. [Analysis](#Analysis)
5. [Results](#Results)
6. [Future Improvements](#FutureImprovements)
7. [Licensing, Authors, and Acknowledgements](#Licensing)
8. [Instructions](#Instructions)


## Libraries <a name="Libraries"></a>
* pandas
* numpy
* sqlalchemy
* re
* plotly
* NLTK
* pickle
* NLTK [punkt, wordnet, stopwords]
* sklearn
* flask

## Project Description <a name="ProjectDescription"></a>
Appen or previously named Figure Eight Data Set: provides thousands of labeled messages. These messages are sorted into specific categories such as Water, Medical-Aid, Aid-Related, that are specifically aimed at helping emergency personnel in their efforts.

The main goal of this project is to build an web app that can help emergency personnel analyze incoming messages and sort them into specific categories to speed up the aid and be more efficient to be able to help more people in need.

## File Description <a name="FileDescription"></a>
There are three main folders:

1. data
    - disaster_categories.csv: dataset containing the categories
    - disaster_messages.csv: dataset containing the messages
    - process_data.py: ETL pipeline scripts to read, clean, and save data into a database
    - DisasterResponse.db: the ETL pipeline output, (SQLite database) containing messages and categories data
    
2. app
    - run.py: Flask file to run the web application
    - templates containing html files for the web application
    
3. models
    - train_classifier.py: machine learning pipeline scripts to train and save the model classifier
    - classifier.pkl: saved output of the machine learning pipeline


## Results <a name="Results"></a>
1. Created an ETL pipeline to read data from csv files, cleaning it, and saving the data into a SQLite database.
2. Created a machine learning pipeline to train a multi-output classifier on the different categories in the dataset.
3. Created a Flask app to show visualization and classify any user input messages on the web app.


## Licensing, Authors, and Acknowledgements <a name="Licensing"></a>
Thanks to Udacity for the starter code and Appen (figure Eight) for providing the data set to be used in this project.


## Instructions <a name="Instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves model
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

![alt text](https://github.com/navidebra/drsproj/blob/main/Images/Screenshot%202022-08-22%20at%2022-47-35%20Disasters.png)
![alt text](https://github.com/navidebra/drsproj/blob/main/Images/Screenshot%202022-08-22%20at%2022-52-27%20Disasters.png)

