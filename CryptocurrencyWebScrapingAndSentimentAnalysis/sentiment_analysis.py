import numpy as np
import math
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA


def analyse_text(text, keywords):
    sia = SIA()

    sentiment_out = 0
    no_keyword_counter = 0
    for keyword in keywords:
        if keyword in text:
            sentiment = sia.polarity_scores(text)

            sentiment = sentiment['compound']
            sentiment_out += sentiment
        else:
            no_keyword_counter += 1

    if no_keyword_counter == len(keywords):
        sentiment_out = np.nan

    return sentiment_out


def analyse_sentiments(dates_local, texts_local, keywords):
    sentiments_local = []
    dates_new = []
    texts_new = []
    sentiments_new = []

    for i in range(len(texts_local)):
        sentiments_local.append(analyse_text(texts_local[i], keywords))

        if not math.isnan(sentiments_local[i]):
            dates_new.append(dates_local[i])
            texts_new.append(texts_local[i])
            sentiments_new.append(sentiments_local[i])

    return dates_new, texts_new, sentiments_new
