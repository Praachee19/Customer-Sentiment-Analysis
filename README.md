import numpy as np
import pandas as pd

df = pd.read_csv('/content/Womens Clothing E-Commerce Reviews.csv')

df.head()

Explore The data

df.shape

df.info()

df.isnull().sum()

df.dropna(inplace=True)

df['Rating'].unique()

import plotly.express as px

rating_counts = df['Rating'].value_counts().reset_index()
rating_counts.columns = ['Rating', 'count']

fig = px.bar(rating_counts, x='Rating', y='count', title='Distribution of Ratings')
fig.show()

df.shape

print("Number of duplicate rows: ", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Number of rows after removing duplicates: ", len(df))

df.drop(['Clothing ID', 'Unnamed: 0'], axis=1, inplace=True)

import plotly.express as px

class_counts = df['Class Name'].value_counts().reset_index()
class_counts.columns = ['Class Name', 'count']

fig = px.bar(class_counts, x='Class Name', y='count', title='Distribution of Class')
fig.show()

import plotly.express as px

class_counts = df['Department Name'].value_counts().reset_index()
class_counts.columns = ['Department Name', 'count']

fig = px.bar(class_counts, x='Department Name', y='count', title='Distribution of Department')
fig.show()

import plotly.express as px

class_counts = df['Division Name'].value_counts().reset_index()
class_counts.columns = ['Division Name', 'count']

fig = px.bar(class_counts, x='Division Name', y='count', title='Distribution of Division')
fig.show()

df.columns

Text Processing - removing stop words, punctuations

df['Review Text'] = df['Review Text'].str.lower()

import re
import nltk
from nltk.corpus import stopwords
import string

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

def remove_stopwords_punctuation(text):
    try:
        text = str(text) # Ensure it's a string
        text = text.translate(str.maketrans('', '', string.punctuation))
        return ' '.join([word for word in text.split() if word not in stop_words])
    except Exception as e:
        print(f"Error processing text: {text}, Type: {type(text)}, Error: {e}")
        return '' # Return empty string or handle as needed

df['Review Text'] = df['Review Text'].astype(str).apply(remove_stopwords_punctuation)

import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def stem_text(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])

def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

df['Review Text_stemmed'] = df['Review Text'].apply(stem_text)
df['Review Text_lemmatized'] = df['Review Text'].apply(lemmatize_text)

display(df[['Review Text', 'Review Text_stemmed', 'Review Text_lemmatized']].head())

import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab') # Download the missing resource

def tokenize_text(text):
    return word_tokenize(text)

df['Review Text_tokens'] = df['Review Text'].apply(tokenize_text)

display(df[['Review Text', 'Review Text_tokens']].head())

### Text-based Exploration - Sentiment & Recommendation Classification

Generate word cloud for neutral ratings (Rating 3)



df_neutral_ratings = df[df['Rating'] == 3]

print("Shape of df_neutral_ratings:", df_neutral_ratings.shape)

neutral_ratings_text = ' '.join(df_neutral_ratings['Review Text'])

print("Length of neutral_ratings_text:", len(neutral_ratings_text))

from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud_neutral = WordCloud(width=800, height=400, background_color='white').generate(neutral_ratings_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_neutral, interpolation='bilinear')
plt.axis('off')
plt.show()


Filter the DataFrame to create two groups based on ratings: one for ratings 1 and 2, and another for ratings 4 and 5.



df_low_ratings = df[df['Rating'].isin([1, 2])]
df_high_ratings = df[df['Rating'].isin([4, 5])]

print("Shape of df_low_ratings:", df_low_ratings.shape)
print("Shape of df_high_ratings:", df_high_ratings.shape)

low_ratings_text = ' '.join(df_low_ratings['Review Text'])
high_ratings_text = ' '.join(df_high_ratings['Review Text'])

print("Length of low_ratings_text:", len(low_ratings_text))
print("Length of high_ratings_text:", len(high_ratings_text))

from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud_high = WordCloud(width=800, height=400, background_color='white').generate(high_ratings_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_high, interpolation='bilinear')
plt.axis('off')
plt.show()

from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud_low = WordCloud(width=800, height=400, background_color='white').generate(low_ratings_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_low, interpolation='bilinear')
plt.axis('off')
plt.show()

## Summary:



*   The dataset was successfully filtered into two groups: low ratings (1 and 2) containing 2051 reviews and high ratings (4 and 5) containing 15147 reviews.
*   The review text for low ratings was concatenated into a string of 400,139 characters, and the text for high ratings was concatenated into a string of 2,922,271 characters.
*   Word clouds were generated and displayed for both the low-rated and high-rated review text, visually representing the most frequent words in each category.




Comparing the Word Clouds:

Look for prominent words in each cloud: Identify the largest and most frequently appearing words in the word cloud for low ratings and compare them to the largest words in the word cloud for high ratings. These prominent words are likely key indicators of customer sentiment.
Identify unique words: Look for words that appear in one word cloud but are absent or much smaller in the other. These unique terms can highlight specific aspects that differentiate positive and negative experiences.

Consider the context of the words: Think about the context in which these words are typically used in reviews. For example, words like "small," "tight," or "cheap" in the low ratings word cloud might indicate issues with sizing or quality, while words like "love," "great," or "perfect" in the high ratings word cloud suggest satisfaction with the product or fit.
Interpreting the Differences:

Low Ratings: Words that are large and frequent in the low ratings word cloud likely represent common complaints or negative experiences. Pay attention to terms related to fit, material, quality, or discrepancies between the product description and the actual item.
High Ratings: Words that are large and frequent in the high ratings word cloud likely represent aspects that customers appreciate and lead to positive experiences. Look for terms related to fit, comfort, style, quality, or value for money.
By carefully examining the words and their relative sizes in both word clouds, you can gain valuable insights into the key drivers of customer satisfaction and dissatisfaction.

Convert Rating to Sentiment and Remove Neutral Reviews

# Function to map ratings to sentiment labels
def get_sentiment(rating):
    if rating in [4, 5]:
        return 'Positive'
    elif rating in [1, 2]:
        return 'Negative'
    else:
        return None # Mark neutral ratings to be removed

# Apply the function to create the 'Sentiment' column
df['Sentiment'] = df['Rating'].apply(get_sentiment)

# Remove rows with neutral ratings (where Sentiment is None)
df.dropna(subset=['Sentiment'], inplace=True)

# Display the counts of each sentiment to verify
display(df['Sentiment'].value_counts())

# Display the head of the dataframe with the new 'Sentiment' column
display(df[['Rating', 'Sentiment']].head())

### TF-IDF Embedding - Predicting Recommendation

Importing the libraries

!pip install matplotlib-venn
# https://pypi.python.org/pypi/libarchive
!apt-get -qq install -y libarchive-dev && pip install -U libarchive
import libarchive
# https://pypi.python.org/pypi/pydot
!apt-get -qq install -y graphviz && pip install pydot
import pydot
!pip install cartopy
import cartopy

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000) # You can adjust max_features

# Fit and transform the 'Review Text' column
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Review Text'])

# Display the shape of the TF-IDF matrix
print("Shape of TF-IDF matrix:", tfidf_matrix.shape)

!pip install nltk
import nltk
nltk.download('vader_lexicon')

%pip install vaderSentiment

!pip install matplotlib-venn

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('punkt') # Required for TextBlob
nltk.download('averaged_perceptron_tagger') # Required for TextBlob

analyzer = SentimentIntensityAnalyzer()

def get_textblob_polarity(text):
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return 0.0 # Return 0 for errors

df['TextBlob_Polarity'] = df['Review Text'].apply(get_textblob_polarity)

display(df[['Review Text', 'TextBlob_Polarity']].head())

def get_vader_sentiment(text):
    try:
        # Ensure text is a string
        text = str(text)
        return analyzer.polarity_scores(text)['compound']
    except:
        return 0.0 # Return 0 for errors

df['VADER_Compound'] = df['Review Text'].apply(get_vader_sentiment)

display(df[['Review Text', 'VADER_Compound']].head())

%pip install textblob vaderSentiment

Prepare data for modeling


X = df[['VADER_Compound']] # Features (VADER compound score)
y = df['Sentiment'] # Target variable (Sentiment)

print("Shape of features (X):", X.shape)
print("Shape of target (y):", y.shape)

Split data



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

Build and train Logistic Regression model


from sklearn.linear_model import LogisticRegression

# Initialize Logistic Regression model
logreg_model = LogisticRegression()

# Train the model
logreg_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_logreg = logreg_model.predict(X_test)

 Evaluate Logistic Regression model



from sklearn.metrics import accuracy_score

# Calculate accuracy
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)

print("Accuracy of Logistic Regression model:", accuracy_logreg)

# Build Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Build and train Random Forest model
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_rf = random_forest_model.predict(X_test)

# Evaluate Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Accuracy of Random Forest model:", accuracy_rf)

df['clean_reviews'] = df['Review Text_tokens'].apply(lambda tokens: ' '.join(tokens))
display(df[['Review Text_tokens', 'clean_reviews']].head())

# Randon Forest

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define features (X) and target (y)
X = df['clean_reviews']
y = df['Recommended IND']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000) # You can adjust max_features

# Fit and transform the training data, transform the test data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print("Shape of X_train_tfidf:", X_train_tfidf.shape)
print("Shape of X_test_tfidf:", X_test_tfidf.shape)

# Build and train Random Forest model
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train_tfidf, y_train)

# Make predictions on the test data
y_pred_rf = random_forest_model.predict(X_test_tfidf)

# Evaluate Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Accuracy of Random Forest model:", accuracy_rf)

# Naive Bayes

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Build and train Naive Bayes model
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train_tfidf, y_train)

# Make predictions on the test data
y_pred_nb = naive_bayes_model.predict(X_test_tfidf)

# Evaluate Naive Bayes model
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Accuracy of Naive Bayes model:", accuracy_nb)

from sklearn.metrics import classification_report

# Classification report for Random Forest model
print("Classification Report for Random Forest Model:")
print(classification_report(y_test, y_pred_rf))

# Classification report for Naive Bayes model
print("\nClassification Report for Naive Bayes Model:")
print(classification_report(y_test, y_pred_nb))

comaparision

Overall Accuracy: Random Forest and Logistic Regression have similar overall accuracy, slightly higher than Naive Bayes.
Performance on 'Recommended' (Class 1): All three models perform very well in predicting recommended items, with high precision, recall, and F1-scores. Logistic Regression and Naive Bayes have a recall of 1.00, meaning they correctly identify all recommended items in the test set, while Random Forest has a recall of 0.99.
Performance on 'Not Recommended' (Class 0): This is where the models differ more and where they struggle due to the data imbalance. Logistic Regression has the best balance of precision (0.88) and recall (0.48) for this class, resulting in a higher F1-score (0.62) compared to Random Forest (F1-score: 0.32) and Naive Bayes (F1-score: 0.18).
In summary, while all models are good at identifying recommended items, Logistic Regression is better at identifying not recommended items compared to Random Forest and Naive Bayes, making it the best performing model among the three for this dataset and evaluation metrics.
