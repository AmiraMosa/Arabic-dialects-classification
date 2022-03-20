# Arabic-dialects-classification
> Predict the Arabic dialect of a given tweet using machine learning and deepl learning models through a REST API
## Table of contents:
* [Dataset](#Dataset)
* [Machine learning model](#Machine-learning-model)
* [Deep learning model](#Deep-learning-model)
* [Models deployments](#Models-deployment)
* [Tools](#Tools)


## Dataset:
- given a dialect_dataset.csv file containing (id-dialect) columns, The “id” column is used to retrieve the tweets, to do that, I call this API  https://recruitment.aimtechnologies.co/ai-tasks by a POST request in dataFetching.py
- save new dataset with tweets in arabic_tweets.csv
- finally save clean dataset in 'clean_data.csv' file


## Machine-learning-model:
Logistic regression

## Deepl-learning-model:
LSTM

## Models-deployment:
FastAPI

## Tools:
- python 3.8
- Keras
- FastAPI
