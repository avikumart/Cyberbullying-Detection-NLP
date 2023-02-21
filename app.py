# import python libraries
import pandas as pd
import streamlit as st
import re
import nltk
from nltk import word_tokenize
import spacy
nlp = spacy.load("en_core_web_sm")

file_path = 'raw_text_file.csv'

st.set_page_config(page_title="Cyberbullying detection app",
                page_icon="👀", layout="wide")


@st.cache
def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df

# text cleaning function
only_english = set(nltk.corpus.words.words())
def text_cleaning(text):
    """ A function to clean the text data"""
    
    sample = text
    sample = " ".join([x.lower() for x in sample.split()])
    sample = re.sub(r"\S*https?:\S*", '', sample) # links and urls
    sample = re.sub('\[.*?\]', '', sample) #text between [square brackets]
    sample = re.sub('\(.*?\)', '', sample) #text between (parenthesis)
    sample = re.sub('[%s]' % re.escape(string.punctuation), '', sample) #punctuations
    sample = re.sub('\w*\d\w', '', sample) #digits with trailing or preceeding text
    sample = re.sub(r'\n', ' ', sample) #new line character
    sample = re.sub(r'\\n', ' ', sample) #new line character
    sample = re.sub("[''""...“”‘’…]", '', sample) #list of quotation marks
    sample = re.sub(r', /<[^>]+>/', '', sample)    #HTML attributes
    
    sample = ' '.join([w for w in nltk.wordpunct_tokenize(sample) if w.lower() in only_english or not w.isalpha()]) #doesn't remove indian languages
    sample = ' '.join(list(filter(lambda ele: re.search("[a-zA-Z\s]+", ele) is not None, sample.split()))) #languages other than english
    
    sample = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE).sub(r'', sample) #emojis and symbols
    sample = sample.strip()
    sample = " ".join([x.strip() for x in sample.split()])
    
    return sample


# pos display
def sentense_pos_tagger(text):
    # tokenize the text
    sample = word_tokenize(text)
    tags = nltk.pos_tag(sample)
        
    return tags

# Front end web UI
st.markdown("<h1 style='text-align: center;'>Cyberbullying detection app 👀</h1>", unsafe_allow_html=True)

# post the image of the cyberbuylling illustration
a,b,c = st.columns([0.2,0.6,0.2])
with b:
    st.image("cyber-bullying.jpeg", use_column_width=True)

# description about the project and code files            
st.subheader("🧾Description:")
st.text("""This dataset is a collection of datasets from different sources related to the automatic detection
of cyber-bullying. The data is from different social media platforms like Kaggle, Twitter, Wikipedia 
Talk pages, and YouTube. The data contains text and are labeled as bullying or not. 
The data contains different types of cyber-bullying like hate speech, aggression, insults, and 
toxicity. You have been provided with the twitter_parsed tweets dataset, 
wherein you have to classify whether the tweet is toxic or not.
""")

st.markdown("Source of the dataset: [Click Here](https://www.kaggle.com/datasets/saurabhshahane/cyberbullying-dataset)")

st.subheader("🧭 Problem Statement:")
st.text("""You are provided with twitter_parsed_tweets: you have to perform a step-by-step NLP approach 
to identify the toxicity of the tweet, and classify the tweet in a binary value.

The target variable is `oh-label` and the evaluation metric is `F1-score`.
""")

st.markdown("Please find GitHub repository link of project: [Click Here](https://github.com/avikumart/Cyberbullying-Detection-NLP)")

# text input from user
text_input = st.text_input("Enter the sample record of tweet", "A sample tweet")

ndf = read_csv(file_path)

# take input between maximum number of rows of dataframe
select_num = st.number_input("Select number of first rows to choose for Named entity display",
                             min_value=10, max_value=16000)


# NER display
def ner_df(n):
    # identify the text object
    doc = nlp(",".join([te for te in ndf['cleaned_text'][0:n]]))

    ner_list = [(x.text,x.label_) for x in doc.ents]
    
    ner_df = pd.DataFrame(ner_list, columns=['Word','Entity identification'])
    return ner_df

# functions display display
def main():
    text = text_cleaning(text_input)
    tags = sentense_pos_tagger(text)
    df = pd.DataFrame(tags, columns=['word','tag'])
    # display dataframe
    st.dataframe(df)
    
    # NER run
    new_df = ner_df(select_num)
    st.dataframe(new_df)
        
if __name__ == '__main__':
    main()


    
    
# topic modeling display



# text classification model inference