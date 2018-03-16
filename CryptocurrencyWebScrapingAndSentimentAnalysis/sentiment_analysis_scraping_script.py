from web_scraper import scrape_subreddits, scrape_forums
from sentiment_analysis import analyse_sentiments

from matplotlib import pyplot as plt


if __name__ == '__main__':
    bitcoin_keywords = ['bitcoin', 'bitcoins', 'xbt', 'btc', 'Bitcoin', 'Bitcoins', 'BTC', 'XBT']
    subreddits = ["cryptocurrency", "cryptomarkets", "bitcoin", "bitcoinmarkets"]

    forum_urls = ["https://bitcointalk.org/index.php?board=5.0", "https://bitcointalk.org/index.php?board=7.0",
                  "https://bitcointalk.org/index.php?board=8.0"]
    allowed_domains = ["bitcointalk.org",]

    dates, texts = scrape_subreddits(subreddits, submission_limit=20)

    dates_temp, texts_temp = scrape_forums(forum_urls, allowed_domains, max_pages=20)

    dates += dates_temp
    texts += texts_temp

    dates, texts, sentiments = analyse_sentiments(dates, texts, bitcoin_keywords)

    plt.figure(1)
    plt.plot(dates, sentiments, "o")
    plt.show()
