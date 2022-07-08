# Disaster Response Pipelines

## Table of Contents

* [Project Overview](#project-overview)
* [Requirements](#requirements)
* [File Descriptions](#file-descriptions)
* [Demo](#demo)
* [Acknowledgements](#acknowledgements)

## Project Overview

In this project, I applied my data engineering skills to build a complete pipeline for data processing and model training for disaster text data classification. In addition, the project also provides a web interface for users to use conveniently. The dataset can be found [here](https://github.com/thekael99/project_2/tree/master/data).

ML model pipeline can be found in [here](https://github.com/thekael99/project_2/tree/master/models).

UI [web app](https://github.com/thekael99/project_2/tree/master/app) to read input and classify  a message into several categories.

## Requirements

This project should be run with these following libraries

* numpy
* pandas
* nltk
* sklearn
* flask
* sqlalchemy
* pickle

## File Descriptions

This project has `3 folders`:

1. `data`: That contains
    * Python script "process_data.py" to process data
    * 02 data files: disaster_categories.csv, disaster_messages.csv

2. `models`: That contains
    * Python script "train_classifier.py" to run ML pipeline

3. `app`: That contains
    * a web app flask.

## Demo

1. Run the following commands in the project's root folder.

    * Run ETL pipeline to clean data and stores in database:

        `python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

    * Run ML pipeline to train and save classifier model:

        `python3 models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Access `app` directory: `cd app`

3. Run web app: `python3 run.py`

## Acknowledgements

* Good project to overview ML pipeline
