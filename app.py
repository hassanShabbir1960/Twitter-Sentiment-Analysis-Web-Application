import os
import numpy as np
import pandas as pd
import GetOldTweets3 as got
from sklearn.utils import shuffle
from flask import Flask, request, jsonify, render_template, redirect
from tensorflow.keras.models import model_from_json
import pickle
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import preprocess

class EmotionAnalysisApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.route('/', methods=['GET', 'POST'])(self.index)

    def fetch_tweets(self, hashtag, s_date, e_date, tweetcount):
        """
        Fetches tweets based on hashtag, date range and tweet count.

        Parameters:
        ----------
        hashtag : str
            The hashtag to search for in the tweets.
        s_date : str
            The start date for the tweet search in the format YYYY-MM-DD.
        e_date : str
            The end date for the tweet search in the format YYYY-MM-DD.
        tweetcount : int
            The number of tweets to fetch.

        Returns:
        -------
        arr : list
            The list of tweets fetched based on the given parameters.
        """
        # Query to fetch tweets
        print("Fetching tweets")
        tweetCriteria = got.manager.TweetCriteria().setQuerySearch(hashtag).setSince(s_date).setUntil(e_date).setMaxTweets(tweetcount)
        # Storing fetched tweets in an array
        arr = [got.manager.TweetManager.getTweets(tweetCriteria)[i].text for i in range(tweetcount)]

        return arr

    def create_dataframe(self, arr, tweetcount):
       
        """
        Creates a dataframe from the fetched tweets and preprocesses it.

        Parameters:
        ----------
        arr : list
            The list of tweets fetched using the fetch_tweets function.
        tweetcount : int
            The number of tweets to include in the dataframe.

        Returns:
        -------
        df : pandas.DataFrame
            The preprocessed dataframe containing the tweets.
        """

        df = pd.DataFrame()
        df['text'] = arr
        df = shuffle(df)  # Shuffling the extracted tweets
        df = df[df['text'].map(len) > 0]  # removing rows with tweets of length 0
        df = df[0:tweetcount]  # Getting first n tweets

        return df

    def emotion_analysis(self, df):
        """
        Performs emotion analysis on the given dataframe and saves the results in a csv file.

        Parameters:
        ----------
        df : pandas.DataFrame
            The preprocessed dataframe containing the tweets.

        Returns:
        -------
        None
        """

        # Loading tokenizer model
        with open('./static/data/tokenizer.pickle', 'rb') as handle:
            tokenizer_obj = pickle.load(handle)

        # Getting scraped tweets
        test_samples = list(df['text'].values)

        # Converting it to required shape
        test_samples_tokens = tokenizer_obj.texts_to_sequences(test_samples)
        test_samples_tokens_pad = pad_sequences(test_samples_tokens, maxlen=300)

        # Loading saved model
        model = load_model('./static/data/my_model.h5')

        # Predict probabilities
        probs = np.round(model.predict_proba(x=test_samples_tokens_pad), 4) * 100

        results = pd.DataFrame(probs[:, 1:], columns=["JOY", "FEAR", "ANGER", "SADNESS", "DISGUST", "SHAME", "GUILT"])
        results['text'] = test_samples
        results['class'] = model.predict_classes(test_samples_tokens_pad)
        results['class'] = results['class'].map({1: "JOY", 2: "FEAR", 3: "ANGER", 4: "SADNESS", 5: "DISGUST", 6: "SHAME", 7: "GUILT"})

        if not os.path.isfile("EmotionAnalysis.csv"):
            print("Making new file")
            results.to_csv("EmotionAnalysis.csv", index=None)
        else:
            print("Changing the existing file")
            df2 = pd.read_csv("EmotionAnalysis.csv")
            df2 = pd.concat([df2, results], axis=0)
            df2.to_csv("EmotionAnalysis.csv", index=None)

        print("Done, Results saved for model 1")

    def sentiment_analysis(self, df):
        """
        Performs sentiment analysis on the given dataframe and saves the results in a csv file.

        Parameters:
        ----------
        df : pandas.DataFrame
            The preprocessed dataframe containing the tweets.

        Returns:
        -------
        None
        """

        # Loading the saved model
        with open('./static/data/sentimentmodel.pkl', 'rb') as f:
            clf = pickle.load(f)
            df['normalized'] = preprocess.preprocess_csv(df['text'].values)
        vectorizer = pickle.load(open("./static/data/vectorizer.pickle", "rb"))
        tweets = df['text'].values
        processed_tweets = df['normalized'].values
        temp = vectorizer.transform(processed_tweets).toarray()

        # Predicting probabilities
        probs = np.round(clf.predict_proba(temp), 4) * 100

        results2 = pd.DataFrame(probs, columns=["neutral", "negative", "positive"])
        results2['text'] = tweets
        results2['class'] = clf.predict(temp)
        results2['class'] = results2['class'].map({1: "negative", 0: "neutral", 2: "positive"})

        if not os.path.isfile("SentimentAnalysis.csv"):
            print("Making new file")
            results2.to_csv("SentimentAnalysis.csv", index=None)
        else:
            print("Changing the existing file")
            df4 = pd.read_csv("SentimentAnalysis.csv")
            df4 = pd.concat([df4, results2], axis=0)
            df4.to_csv("SentimentAnalysis.csv", index=None)

        print("Done, Results saved for model 2")

    def index(self):
        """
        Defines the Flask route for the application.

        Parameters:
        ----------
        None

        Returns:
        -------
        render_template : str
            The rendered HTML template containing the results of the analysis.
        """

        errors = []
        results = {}
        if request.method == "POST":
            # Get values from text boxes
            vals = request.values
            hashtag = vals['a']
            tweetcount = int(vals['b'])
            s_date = vals['c']
            e_date = vals['d']

            # Formatting dates
            s_date = s_date.split('-')
            e_date = e_date.split('-')
            s = str(s_date[0] + "-" + s_date[1] + "-" + s_date[2])
            e = str(e_date[0] + "-" + e_date[1] + "-" + e_date[2])

            # Fetch tweets and create dataframe
            arr = self.fetch_tweets(hashtag, s, e, tweetcount)
            df = self.create_dataframe(arr, tweetcount)

            # Perform emotion and sentiment analysis
            self.emotion_analysis(df)
            self.sentiment_analysis(df)

        return render_template('index.html', errors=errors, results=results)

    def run(self, debug=True):
        """
        Runs the Flask application.

        Parameters:
        ----------
        debug : bool, optional
            Whether to run the application in debug mode. Default is False.

        Returns:
        -------
        None
        """

        self.app.run(debug=debug)

            

if __name__ == "__main__":
    app = EmotionAnalysisApp()
    app.run(debug=True)
