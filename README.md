# Cyberbullying-Detection-NLP
In this project, I have analyzed textual dataset to detect cyberbullying on twitter 

### **🧾Description:** 
This dataset is a collection of datasets from different sources related to the automatic detection of cyber-bullying. The data is from different social media platforms like Kaggle, Twitter, Wikipedia Talk pages, and YouTube. The data contains text and are labeled as bullying or not. The data contains different types of cyber-bullying like hate speech, aggression, insults, and toxicity. You have been provided with the twitter_parsed tweets dataset, wherein you have to classify whether the tweet is toxic or not.

**Source of dataset & Data dictionary:** [Click Here](https://www.kaggle.com/datasets/saurabhshahane/cyberbullying-dataset)

### **🧭 Problem Statement:** 
You are provided with twitter_parsed_tweets: you have to perform a step-by-step NLP approach to identify the toxicity of the tweet, and classify the tweet in a binary value. 
- The target variable is `oh-label` and the evaluation metric is `F1-score.`

### Web application

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cyberbullying-detection.streamlit.app/)

### Published article on Analytics Vidhya:

[Click Here](https://www.analyticsvidhya.com/blog/2023/03/detect-cyberbullying-using-topic-modeling-and-sentiment-analysis/)

### **Steps taken to solve the problem**

1. Data loading and understanding to identify statistical properties of the dataset
2. Text data pre-processing using data clearning and text correction techniques
3. Text data visualization
4. Parts-of-tagging on sentences of the dataset
5. Topic modeling by each category of text to identify most discussed topics in text corpus
6. Named-entity recognition in given text corpus
7. Text classification using Tf-idf vectorization and MultinomialNB modeling -> logloss - 8.4
8. Word2Vec embedding and XGBclassifer modeling -> logloss - 12.0

### **Tools used**

1. NLTK
2. GENSIM
3. Spacy
4. TextBlob
5. Pandas, Matplotlib and Scikit-learn

### Acknowledgement:

The Machine Learning Company
