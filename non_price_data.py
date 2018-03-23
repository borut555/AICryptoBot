import time
import json
import numpy as np
from pytrends.request import TrendReq
from blockchain import util
from API_settings import google_username, google_password


def initialise_google_session():
    return TrendReq(google_username, google_password)


def google_trends_interest_over_time(pytrend_local, search_terms):
    pytrend_local.build_payload(kw_list=search_terms)

    interest_time_df = pytrend_local.interest_over_time()

    unix_times = convert_timestamp_to_unix_time(interest_time_df[search_terms[0]])

    return unix_times, interest_time_df[search_terms[0]].tolist()


def hash_rate():
    response = util.call_api('charts/hash-rate?format=json', base_url='https://api.blockchain.info/')
    hash_json = json.loads(response)

    times = np.zeros(len(hash_json['values']))
    hash_rates = np.zeros(len(hash_json['values']))
    for i in range(len(hash_json['values'])):
        times[i] = hash_json['values'][i]['x']
        hash_rates[i] = hash_json['values'][i]['y']

    return times, hash_rates

def convert_timestamp_to_unix_time(timestamps):
    unix_times = []
    for i in range(len(timestamps.index)):
        unix_times.append(time.mktime(list(timestamps.index)[i].timetuple()))

    return unix_times
