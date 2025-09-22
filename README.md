# Tweet Sentiment Classification

Objective: Classify tweets as positive or negative using NLP preprocessing and ML classifiers.

How to run:
1. Create & activate venv
2. pip install -r requirements.txt
3. jupyter lab
4. Open `tweet_sentiment.ipynb` (select kernel "Python (tweets_venv)") and run cells.

Data:
- If you have your own CSV, place it at data/tweets.csv with columns 'text' and 'label' (label: 1 positive, 0 negative).
- Otherwise notebook will use NLTK's twitter_samples dataset automatically.
