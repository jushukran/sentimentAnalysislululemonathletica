import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from collections import Counter
import re

# Ensure you've downloaded the necessary NLTK resources
nltk.download('vader_lexicon')

# Load reviews from a CSV file
df = pd.read_csv('/lululemon_reviews.csv')

# Check if the 'Review' column exists
if 'Review' not in df.columns:
    raise ValueError("CSV file must contain a 'Review' column.")

# Initialize NLTK's VADER sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()

# Analyzing the sentiment of each review in the dataframe
df['Sentiment Scores'] = df['Review'].apply(lambda review: sia.polarity_scores(review))
df['Compound Score'] = df['Sentiment Scores'].apply(lambda scores: scores['compound'])

# Add labels based on compound score
df['Sentiment Label'] = df['Compound Score'].apply(lambda score: 'Positive' if score >= 0.05 else ('Negative' if score <= -0.05 else 'Neutral'))

# Save the results back to a new CSV file with labels
df.to_csv('sentiment_analysis_with_labels.csv', index=False)

# Print the reviews with their corresponding sentiment scores and labels
print(df[['Review', 'Sentiment Scores', 'Compound Score', 'Sentiment Label']])

# Plotting the Sentiment Distribution
sentiment_counts = df['Sentiment Label'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, colors=['green', 'red', 'blue'])
plt.title('Sentiment Distribution of Lululemon Reviews')
plt.show()

# Find most frequently repeated sentiments suggesting product improvements
negative_reviews = df[df['Sentiment Label'] == 'Negative']['Review']

# Tokenize and clean the text
tokens = [word for review in negative_reviews for word in re.findall(r'\b\w+\b', review.lower())]
common_words = Counter(tokens).most_common(10)

print("Most common words in negative reviews suggesting product improvements:")
for word, freq in common_words:
    print(f"{word}: {freq}")
