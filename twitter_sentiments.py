# Import necessary libraries
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from wordcloud import WordCloud
from collections import Counter

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load datasets
train = pd.read_csv('train.csv')  # Training dataset
test = pd.read_csv('test_tweets.csv')  # Test dataset

# Combine train and test datasets for preprocessing
combi = pd.concat([train, test], ignore_index=True)

# Function to remove unwanted text patterns
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt

# Remove Twitter handles (@user)
combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*")

# Remove punctuations, numbers, and special characters
combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")

# Remove short words (e.g., 'hi', 'is', etc.)
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))

# Tokenization and Lemmatization
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split())  # Tokenizing
lemmatizer = WordNetLemmatizer()

# Lemmatizing each token
tokenized_tweet = tokenized_tweet.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])

# Stitching tokens back together
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

combi['tidy_tweet'] = tokenized_tweet

# Visualizations

# Non-racist/sexist tweets
normal_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 0]])

wordcloud_normal = WordCloud(width=800, height=500, background_color='white', colormap='viridis', random_state=21, max_font_size=110).generate(normal_words)

plt.figure(figsize=(12,6))
plt.imshow(wordcloud_normal, interpolation="bilinear")
plt.axis('off')
plt.title("WordCloud for Non-Racist/Sexist Tweets", fontsize=16)
plt.show()

# Racist/sexist tweets
negative_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 1]])

wordcloud_negative = WordCloud(width=800, height=500, background_color='white', colormap='inferno', random_state=21, max_font_size=110).generate(negative_words)

plt.figure(figsize=(12,6))
plt.imshow(wordcloud_negative, interpolation="bilinear")
plt.axis('off')
plt.title("WordCloud for Racist/Sexist Tweets", fontsize=16)
plt.show()

# Hashtag Analysis

# Function to collect hashtags
def extract_hashtags(x):
    hashtags = []
    for tweet in x:
        ht = re.findall(r"#(\w+)", tweet)
        hashtags.extend(ht)
    return hashtags

# Extracting hashtags from non-racist/sexist tweets
HT_regular = extract_hashtags(combi['tidy_tweet'][combi['label'] == 0])

# Extracting hashtags from racist/sexist tweets
HT_negative = extract_hashtags(combi['tidy_tweet'][combi['label'] == 1])

# Plotting top 20 hashtags for non-racist/sexist tweets
ht_regular_counts = Counter(HT_regular).most_common(20)
ht_regular_df = pd.DataFrame(ht_regular_counts, columns=['Hashtag', 'Count'])

plt.figure(figsize=(12,6))
sns.barplot(data=ht_regular_df, x='Hashtag', y='Count', palette='coolwarm')
plt.xticks(rotation=90)
plt.title("Top 20 Hashtags in Non-Racist/Sexist Tweets", fontsize=16)
plt.show()

# Plotting top 20 hashtags for racist/sexist tweets
ht_negative_counts = Counter(HT_negative).most_common(20)
ht_negative_df = pd.DataFrame(ht_negative_counts, columns=['Hashtag', 'Count'])

plt.figure(figsize=(12,6))
sns.barplot(data=ht_negative_df, x='Hashtag', y='Count', palette='coolwarm')
plt.xticks(rotation=90)
plt.title("Top 20 Hashtags in Racist/Sexist Tweets", fontsize=16)
plt.show()

# Feature Extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])

# Splitting data into train and test sets
train_tfidf = tfidf[:train.shape[0]]
test_tfidf = tfidf[train.shape[0]:]

xtrain_tfidf, xvalid_tfidf, ytrain, yvalid = train_test_split(train_tfidf, train['label'], random_state=42, test_size=0.3)

# Model Training - Logistic Regression
lreg = LogisticRegression()
lreg.fit(xtrain_tfidf, ytrain)

# Predictions on validation set
prediction = lreg.predict_proba(xvalid_tfidf)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(int)

# F1 Score
f1 = f1_score(yvalid, prediction_int)
print(f"F1 Score on Validation Set (Logistic Regression): {f1}")

# Confusion Matrix and Classification Report
print(classification_report(yvalid, prediction_int))
sns.heatmap(confusion_matrix(yvalid, prediction_int), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# Model Training - XGBoost
xgb = XGBClassifier(objective='binary:logistic', n_estimators=400, learning_rate=0.1, max_depth=8, subsample=0.9, colsample_bytree=0.5, min_child_weight=6, gamma=1.2, random_state=42)
xgb.fit(xtrain_tfidf, ytrain)

# Predictions on validation set
xgb_prediction = xgb.predict(xvalid_tfidf)
xgb_f1 = f1_score(yvalid, xgb_prediction)
print(f"F1 Score on Validation Set (XGBoost): {xgb_f1}")

# Confusion Matrix and Classification Report
print(classification_report(yvalid, xgb_prediction))
sns.heatmap(confusion_matrix(yvalid, xgb_prediction), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - XGBoost")
plt.show()

# Submission File Creation

# Predictions on test set (Logistic Regression)
test_pred_lr = lreg.predict_proba(test_tfidf)
test_pred_int_lr = test_pred_lr[:,1] >= 0.3
test_pred_int_lr = test_pred_int_lr.astype(int)

submission_lr = pd.DataFrame({'id': test['id'], 'label': test_pred_int_lr})
submission_lr.to_csv('submission_logistic_regression.csv', index=False)

# Predictions on test set (XGBoost)
test_pred_xgb = xgb.predict(test_tfidf)

submission_xgb = pd.DataFrame({'id': test['id'], 'label': test_pred_xgb})
submission_xgb.to_csv('submission_xgboost.csv', index=False)


import os
import matplotlib.pyplot as plt

# Create the 'results/' directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# Example: Word Cloud for Non-Racist/Sexist Tweets
normal_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 0]])

wordcloud_normal = WordCloud(width=800, height=500, background_color='white', colormap='viridis', random_state=21, max_font_size=110).generate(normal_words)

plt.figure(figsize=(12,6))
plt.imshow(wordcloud_normal, interpolation="bilinear")
plt.axis('off')
plt.title("WordCloud for Non-Racist/Sexist Tweets", fontsize=16)

# Save the figure
plt.savefig('results/wordcloud_non_racist.png', bbox_inches='tight', dpi=300)
plt.show()

# Example: Word Cloud for Racist/Sexist Tweets
negative_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 1]])

wordcloud_negative = WordCloud(width=800, height=500, background_color='white', colormap='inferno', random_state=21, max_font_size=110).generate(negative_words)

plt.figure(figsize=(12,6))
plt.imshow(wordcloud_negative, interpolation="bilinear")
plt.axis('off')
plt.title("WordCloud for Racist/Sexist Tweets", fontsize=16)

# Save the figure
plt.savefig('results/wordcloud_racist.png', bbox_inches='tight', dpi=300)
plt.show()

# Example: Top 20 Hashtags in Non-Racist/Sexist Tweets
ht_regular_counts = Counter(HT_regular).most_common(20)
ht_regular_df = pd.DataFrame(ht_regular_counts, columns=['Hashtag', 'Count'])

plt.figure(figsize=(12,6))
sns.barplot(data=ht_regular_df, x='Hashtag', y='Count', palette='coolwarm')
plt.xticks(rotation=90)
plt.title("Top 20 Hashtags in Non-Racist/Sexist Tweets", fontsize=16)

# Save the figure
plt.savefig('results/top_hashtags_non_racist.png', bbox_inches='tight', dpi=300)
plt.show()

# Example: Top 20 Hashtags in Racist/Sexist Tweets
ht_negative_counts = Counter(HT_negative).most_common(20)
ht_negative_df = pd.DataFrame(ht_negative_counts, columns=['Hashtag', 'Count'])

plt.figure(figsize=(12,6))
sns.barplot(data=ht_negative_df, x='Hashtag', y='Count', palette='coolwarm')
plt.xticks(rotation=90)
plt.title("Top 20 Hashtags in Racist/Sexist Tweets", fontsize=16)

# Save the figure
plt.savefig('results/top_hashtags_racist.png', bbox_inches='tight', dpi=300)
plt.show()

# Example: Tweet Length Distribution
length_train = train['tweet'].str.len()
length_test = test['tweet'].str.len()

plt.hist(length_train, bins=20, label="train_tweets", alpha=0.7)
plt.hist(length_test, bins=20, label="test_tweets", alpha=0.7)
plt.legend()
plt.title("Tweet Length Distribution", fontsize=16)

# Save the figure
plt.savefig('results/tweet_length_distribution.png', bbox_inches='tight', dpi=300)
plt.show()