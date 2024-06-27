import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure you've downloaded the necessary NLTK resources
nltk.download('vader_lexicon')

# Load reviews from a CSV file
df = pd.read_csv('/Users/fella/Desktop/Npdp/pythonProject/lululemon_reviews.csv')

# Check if the 'Review' column exists
if 'Review' not in df.columns:
    raise ValueError("CSV file must contain a 'Review' column.")

# Initialize NLTK's VADER sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()

# Analyzing the sentiment of each review in the dataframe
df['Sentiment Scores'] = df['Review'].apply(lambda review: sia.polarity_scores(review))

# Optionally, extract the compound score for further analysis
df['Compound Score'] = df['Sentiment Scores'].apply(lambda scores: scores['compound'])

# Print the reviews with their corresponding sentiment scores
print(df[['Review', 'Sentiment Scores', 'Compound Score']])

# Optionally, save the results back to a new CSV file
df.to_csv('sentiment_analysis_results.csv', index=False)
