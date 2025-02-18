# Twitter Sentiment Analysis: Practice Problem

## Problem Statement

The objective of this project is to detect hate speech in tweets. Specifically, we classify tweets as either containing hate speech (racist or sexist sentiment) or not.

Formally, given a training sample of tweets and labels:
- Label `1` denotes the tweet is racist/sexist.
- Label `0` denotes the tweet is not racist/sexist.

Your task is to predict the labels on the test dataset.

You can access the problem statement and the data [here](https://datahack.analyticsvidhya.com/contest/practice-problem-twitter-sentiment-analysis/).

## Table of Contents

- [Data Exploration](#data-exploration)
- [Text Preprocessing and Cleaning](#text-preprocessing-and-cleaning)
- [Feature Extraction](#feature-extraction)
- [Model Building](#model-building)
- [Model Tuning](#model-tuning)
- [Results](#results)
- [Visualizations](#visualizations)

---

## Data Exploration

### Tweet Length Distribution

The following histogram shows the distribution of tweet lengths in both the training and test datasets. The distributions are quite similar, indicating that the datasets are consistent in terms of tweet length.

![Tweet Length Distribution](tweet_length_distribution.png)

### Label Distribution

In the train dataset:
- 2,242 (~7%) tweets are labeled as racist/sexist.
- 29,720 (~93%) tweets are labeled as non-racist/sexist.

This indicates an imbalanced classification challenge.

---

## Text Preprocessing and Cleaning

We performed several steps to clean and preprocess the text data:

1. **Removing Twitter Handles (@user)**: Removed all mentions of users in the tweets.
2. **Removing Punctuations, Numbers, and Special Characters**: Only alphabetic characters were retained.
3. **Removing Short Words**: Words with less than 3 characters were removed.
4. **Stemming**: Reduced words to their root forms using stemming.

---

## Feature Extraction

### Word Clouds

Word clouds help visualize the most frequent words in the tweets. Larger words indicate higher frequency.

- **Non-Racist/Sexist Tweets Word Cloud**:
  ![Word Cloud Non-Racist](wordcloud_non_racist.png)

- **Racist/Sexist Tweets Word Cloud**:
  ![Word Cloud Racist](wordcloud_racist.png)

### Top Hashtags

Hashtags provide insights into the common topics discussed in the tweets.

- **Top Hashtags in Non-Racist/Sexist Tweets**:
  ![Top Hashtags Non-Racist](top_hashtags_non_racist.png)

- **Top Hashtags in Racist/Sexist Tweets**:
  ![Top Hashtags Racist](top_hashtags_racist.png)

---

## Model Building

We built models using several algorithms:
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

### Feature Sets

We used different feature sets for model training:
- Bag-of-Words (BoW)
- TF-IDF
- Word2Vec Embeddings
- Doc2Vec Embeddings

---

## Model Tuning

We fine-tuned the XGBoost model by optimizing the following hyperparameters:
- `max_depth`
- `min_child_weight`
- `subsample`
- `colsample_bytree`
- `eta`
- `gamma`

---

## Results

### Logistic Regression Model
- **Validation F1 Score**: 0.526
- **Precision**: 0.66
- **Recall**: 0.44
- **Accuracy**: 0.94

### XGBoost Model
- **Validation F1 Score**: 0.417
- **Precision**: 0.72
- **Recall**: 0.29
- **Accuracy**: 0.94

---

## Visualizations

#### Tweet Length Distribution
![Tweet Length Distribution](tweet_length_distribution.png)

#### Word Clouds
- **Non-Racist/Sexist Tweets Word Cloud**:
  ![Word Cloud Non-Racist](wordcloud_non_racist.png)

- **Racist/Sexist Tweets Word Cloud**:
  ![Word Cloud Racist](wordcloud_racist.png)

#### Top Hashtags
- **Top Hashtags in Non-Racist/Sexist Tweets**:
  ![Top Hashtags Non-Racist](top_hashtags_non_racist.png)

- **Top Hashtags in Racist/Sexist Tweets**:
  ![Top Hashtags Racist](top_hashtags_racist.png)

---

## Conclusion

This project demonstrates the process of building and evaluating machine learning models for detecting hate speech in tweets. The visualizations provide insights into the common words and hashtags used in different categories of tweets. The Logistic Regression model performed better compared to the initial XGBoost model on the validation set, but further tuning may improve results.

For more details on the implementation and analysis, refer to the accompanying Jupyter Notebook.
