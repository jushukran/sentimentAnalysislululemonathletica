import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

# Download necessary NLTK resources
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')  # Ensure the punkt tokenizer is downloaded

# Load reviews from a CSV file
df = pd.read_csv('/Users/fella/Desktop/Npdp/pythonProject/.venv/sentiment_analysis_with_labels.csv')

# Initialize NLTK's VADER sentiment intensity analyzer and stop words
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))  # English stop words

# Function to preprocess text by removing stop words and non-alphanumeric characters
def preprocess_text(text):
    words = nltk.word_tokenize(text.lower())  # Tokenize text and convert to lower case
    filtered_words = [word for word in words if word not in stop_words and word.isalnum()]
    return " ".join(filtered_words)

# Apply preprocessing to reviews
df['Processed Review'] = df['Review'].apply(preprocess_text)

# Filter for positive reviews
positive_reviews = df[df['Sentiment Label'] == 'Positive']['Processed Review']

# Combine all positive reviews into a single string
positive_text = " ".join(positive_reviews)

# Generate a word cloud for positive reviews
wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      stopwords=stop_words,
                      min_font_size=10).generate(positive_text)

# Display the word cloud
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

# Optionally, save the word cloud as an image file
plt.savefig('/path/to/your/save_directory/positive_reviews_wordcloud.png')
