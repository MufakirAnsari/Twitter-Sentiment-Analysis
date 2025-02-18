# Twitter Sentiment Analysis: Detecting Racist/Sexist Tweets

This project aims to detect racist and sexist tweets using various machine learning models. The dataset consists of labeled tweets, where the task is to classify them as either racist/sexist or non-racist/non-sexist.

## Table of Contents
- [Dataset Overview](#dataset-overview)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Model Building & Evaluation](#model-building-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)

---

## Dataset Overview
The dataset contains two columns:
- `tidy_tweet`: Cleaned version of the tweet text.
- `label`: Binary label indicating whether a tweet is racist/sexist (`1`) or not (`0`).

---

## Exploratory Data Analysis (EDA)

### Hashtag Analysis
#### Top 20 Hashtags in Non-Racist/Sexist Tweets
![Top 20 Hashtags in Non-Racist/Sexist Tweets](image1.png)

The most common hashtags in non-racist/sexist tweets are positive and neutral terms like `#love`, `#positive`, and `#healthy`.

#### Top 20 Hashtags in Racist/Sexist Tweets
![Top 20 Hashtags in Racist/Sexist Tweets](image2.png)

Racist/sexist tweets contain more politically charged and negative hashtags such as `#trump`, `#politics`, and `#allahsoil`.

---

### Word Clouds
#### WordCloud for Non-Racist/Sexist Tweets
![WordCloud for Non-Racist/Sexist Tweets](image4.png)

Words like "happy," "life," "positive," and "thankful" dominate the word cloud for non-racist/sexist tweets, reflecting their generally positive nature.

#### WordCloud for Racist/Sexist Tweets
![WordCloud for Racist/Sexist Tweets](image5.png)

In contrast, words like "hate," "white," "black," and "liberal" appear frequently in racist/sexist tweets, highlighting the negative and divisive language used.

---

### Tweet Length Distribution
![Tweet Length Distribution](image3.png)

The distribution of tweet lengths shows that both training and test datasets have similar patterns, with most tweets falling within a length range of 50-150 characters.

---

## Feature Engineering
Four types of features were extracted from the tweets:
1. **Bag-of-Words (BoW)**
2. **TF-IDF**
3. **Word2Vec**
4. **Doc2Vec**

Each feature type was used to train different machine learning models.

---

## Model Building & Evaluation

### Logistic Regression
- **Validation F1 Score:** 0.526
- **Classification Report:**
```
              precision    recall  f1-score   support
           0       0.96      0.98      0.97      8905
           1       0.66      0.44      0.53       684
    accuracy                           0.94      9589
   macro avg       0.81      0.71      0.75      9589
weighted avg       0.94      0.94      0.94      9589
```

### XGBoost
- **Validation F1 Score:** 0.417
- **Classification Report:**
```
              precision    recall  f1-score   support
           0       0.95      0.99      0.97      8905
           1       0.72      0.29      0.42       684
    accuracy                           0.94      9589
   macro avg       0.83      0.64      0.69      9589
weighted avg       0.93      0.94      0.93      9589
```

---

## Results
The Logistic Regression model performed better than XGBoost on this dataset. However, further tuning and experimentation with other models could improve performance.

---

## Conclusion
This project demonstrates how machine learning can be applied to detect hate speech on social media platforms. While the initial results are promising, there is still room for improvement through advanced techniques like deep learning and more sophisticated feature engineering.
