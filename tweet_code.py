import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Constants
VIRAL_TWEET_PERCENTILE = 75
RANDOM_STATE = 1

def load_data(file_path):
    """Load data from a JSON file."""
    return pd.read_json(file_path, lines=True)

def prepare_data(df, retweet_threshold):
    """Prepare data for model training."""
    df['is_viral'] = np.where(df['retweet_count'] > retweet_threshold, 1, 0)
    return train_test_split(df["text"], df["is_viral"], random_state=RANDOM_STATE)

def create_and_train_model(train_text, train_labels, test_text):
    """Create and train the model."""
    counter = CountVectorizer()
    counter.fit(pd.concat([train_text, test_text]))
    
    train_text_counts = counter.transform(train_text)
    
    model = MultinomialNB()
    model.fit(train_text_counts, train_labels)
    
    return model, counter

def evaluate_model(model, counter, test_text, test_labels):
    """Evaluate the model."""
    test_text_counts = counter.transform(test_text)
    return model.score(test_text_counts, test_labels)

def predict_tweet(model, counter, tweet):
    """Predict if a tweet is likely to go viral."""
    tweet_counts = counter.transform([tweet])
    prediction = model.predict(tweet_counts)
    probability = model.predict_proba(tweet_counts)
    return prediction, probability

def main():
    # Load data
    file_path = os.path.join(os.path.expanduser("~"), "twitter_classification_project", "random_tweets.json")
    all_tweets = load_data(file_path)

    # Prepare data
    retweet_threshold = np.percentile(all_tweets['retweet_count'], VIRAL_TWEET_PERCENTILE)
    train_text, test_text, train_labels, test_labels = prepare_data(all_tweets, retweet_threshold)

    # Create and train model
    model, counter = create_and_train_model(train_text, train_labels, test_text)

    # Evaluate model
    score = evaluate_model(model, counter, test_text, test_labels)
    print(f"Model score: {score}")

    # Predict new tweet
    new_tweet = "This is a tweet"
    prediction, probability = predict_tweet(model, counter, new_tweet)
    print(f"Prediction: {prediction}")
    print(f"Probability: {probability}")

if __name__ == "__main__":
    main()