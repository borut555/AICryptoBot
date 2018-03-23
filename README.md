# AICryptoBot

AICryptoBot is artifitial intelegent crypto trading algorithm (bot) based on machine learning and deep learning. 
Program is writen in python to generate cryptocurrency trading strategies.
Some ideas and solutions are taken from project https://github.com/owocki/pytrader

## Optimisation of machine learning methods
The optimisation is based on a random search with a bayesian optimization which use hyperparameter fitting for each algorithm. 
The optimizations selects a proper method and a preprocessing pair, the selected method is then optimised using a second random grid search to fit the hyperparameters for that method.

## Validation
Validation of overffiting is performed based on hyperparameter fitting.

## Settings
Set proper values in API_settings.py to enable connection to poloniex crypto exchange, google data, reddit scrapering
poloniex_API_key = ""   
poloniex_API_secret = "" 
google_username = "" 
google_password = "" 
client_id = "" 
client_secret = "" 
user_agent = 'Python Scraping App' 

## Windows
conda create -n tensorflow-p2 python=3.6   
conda activate tensorflow-p2   
conda install numpy pandas matplotlib tensorflow jupyter notebook scipy scikit-learn nb_conda    
conda install -c auto statsmodels   
pip install arch polyaxon
pip instal scrapy, praw, bs4, nltk
pip install keras  
for other missing peckages use pip install
 
