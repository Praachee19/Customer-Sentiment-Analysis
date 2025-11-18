# Customer-Sentiment-Analysis
I explored the Women’s Clothing E Commerce Reviews dataset to understand how customers express satisfaction, disappointment and buying intent. The work spans data cleaning, exploratory analysis, text processing, sentiment scoring and predictive modeling. What follows is a clear walk through of the entire project and the insights that emerged.
Chapter 1. Getting the Data Ready

I start by loading the dataset and checking its structure, missing values and duplicates. This is the foundation. Clean data is non negotiable.

df = pd.read_csv('/content/Womens Clothing E-Commerce Reviews.csv')
df.info()
df.isnull().sum()
df.dropna(inplace=True)
df.duplicated().sum()
df.drop_duplicates(inplace=True)


Columns like Clothing ID and Unnamed: 0 add no analytical value. They are removed to simplify the dataset.

I confirm the basic distributions. Ratings. Department Name. Division Name. Class Name. Visualising them through plotly bar charts quickly shows which segments dominate the dataset.

Chapter 2. Cleaning the Text

Raw review text is messy. I standardise it to lowercase. I strip punctuation. I remove stopwords. This leaves only meaningful terms.

df['Review Text'] = df['Review Text'].str.lower()


Next I apply stemming and lemmatization so that words like “running”, “runs” and “ran” converge to the same form. This makes the modelling later far more efficient.

I then tokenise each review to understand how customers use language.

df['Review Text_tokens'] = df['Review Text'].apply(tokenize_text)


This creates a solid, structured base for sentiment exploration.

Chapter 3. Exploring the Text with Word Clouds

Ratings tell one story. Text tells another. I split the dataset into three groups.

• Low ratings (1 and 2) show dissatisfaction
• Neutral ratings (3) show mixed emotions
• High ratings (4 and 5) show strong satisfaction

I generate three word clouds to visualise the language patterns.

Low rated reviews often highlight sizing issues, poor fit, low quality and design inconsistency.

High rated reviews consistently feature words related to comfort, fit, softness and compliments.

Neutral rated reviews contain mixed and vague descriptive language, indicating uncertainty, hesitation or average experiences.

This visual step is important. It immediately shows what delights customers and what frustrates them.

Chapter 4. Converting Ratings to Sentiment Labels

To simplify modelling, I convert ratings into binary sentiment categories.

Positive
• Ratings 4 and 5

Negative
• Ratings 1 and 2

Neutral ratings are removed because they create ambiguity.

df['Sentiment'] = df['Rating'].apply(get_sentiment)
df.dropna(subset=['Sentiment'], inplace=True)


The dataset becomes a clear two class sentiment system.

Chapter 5. TF IDF Embeddings and Sentiment Scores

I build TF IDF embeddings to capture the importance of words in each review.

tfidf_matrix = tfidf_vectorizer.fit_transform(df['Review Text'])


Along with TF IDF, I compute two sentiment polarity scores.

• TextBlob polarity
• VADER compound score

These numeric indicators reflect the tone of each review. They help later in predictive modelling.

Chapter 6. Predicting Sentiment from VADER Scores

(Logistic Regression and Random Forest)

I create a simple model where the only feature is the VADER compound score. This tests whether a single numeric sentiment score can predict positive or negative customer sentiment.

Logistic Regression performs well. Random Forest performs similarly. This confirms that compound scores already capture strong sentiment signals.

Chapter 7. Predicting Recommendations Using TF IDF

(Random Forest and Naive Bayes)

In the second modelling block, I shift the target from sentiment to recommendation intent.

Target
• Recommended IND (1 or 0)

Feature
• Cleaned review text converted to TF IDF vectors

I train two models.

Random Forest

• Learns complex patterns
• Performs well on high volume text
• Accuracy is strong on class 1
• Struggles slightly on class 0 because the dataset is imbalanced

Naive Bayes

• Lightweight
• Designed for text
• Performs strongly on positive recommendations
• Less capable on non recommended items

Comparative performance

Logistic Regression (from the earlier stage) remains the best at identifying both classes fairly.
Random Forest and Naive Bayes are heavily biased toward predicting recommended items because the dataset contains many more positive recommendations.

Chapter 8. What the Models Reveal
1. Positivity dominates the dataset

High ratings vastly outnumber low ratings. Customers who buy these products generally have good experiences.

2. Negative reviews contain consistent complaint themes

Words such as small, tight, poor, return, cheap and uncomfortable appear frequently. These point to fit and quality as major pain points.

3. Positive reviews focus on fit, comfort and compliments

Words such as love, perfect, soft, flattering and comfortable stand out. These are strong drivers of satisfaction.

4. Logistic Regression delivers the best balance

It identifies non recommended items more accurately than Random Forest or Naive Bayes.
This matters for real world use because retailers care more about identifying unhappy customers early.

Chapter 9. Key Insights for the Business
Fit issues are the biggest source of dissatisfaction

Most negative reviews mention inconsistent sizing, tight fit or incorrect fit descriptions.

Quality and material complaints also contribute to negative sentiment

Thin fabric, poor stitching and fading colors appear frequently.

Positive customers respond to comfort and value

Good fit, flattering cuts, softness and price satisfaction dominate the positive cluster.

Recommendation prediction is feasible

TF IDF based models achieve strong accuracy. They can be deployed to classify incoming reviews automatically and flag reviews needing attention.

Final Summary

Customer sentiment analysis on this dataset reveals clear patterns.

• Positive sentiment is strong and influence by fit, comfort and style
• Negative sentiment revolves around quality problems and inaccurate sizing
• TF IDF embeddings paired with Logistic Regression or Random Forest offer reliable classification
• VADER and TextBlob polarity scores align well with star ratings
• Logistic Regression offers the most balanced model performance
• Retailers can use these insights to improve product descriptions, sizing guides and quality checks

This project demonstrates the full end to end journey of customer sentiment analysis. From data cleaning to modelling to insight extraction. It provides a practical roadmap for understanding what customers actually feel and why they choose to recommend or reject a product.
