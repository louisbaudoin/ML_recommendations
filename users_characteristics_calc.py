# Import dependencies
from __future__ import division
import pandas as pd
import csv
import numpy as np
import datetime #from datetime import datetime
import calendar #necessary to convert a timestamp into a date
import time
from math import sqrt
import numpy.random as random

#Compute correlations
def compute_correlations_users(user_id, data_users):
  #process data
  processed_data_users = process_data_users(data_users)
  # compute correlations
  corr_user = processed_data_users.corrwith(processed_data_users.loc[user_id,:], axis=1, drop=False)
  return corr_user

# dummify, and clean users_data so we can compute correlations
def process_data_users(data_users):
  # dummify
  columns_to_dummify = ['age', 'occupation']
  data_users = _dummify(data_users, columns_to_dummify)
  # drop zip code
  del data_users['zip_code']
  # clean sex
  gender_hash = {'M': 0, 'F': 1}
  data_users['gender'] = data_users['gender'].map(gender_hash)
  return data_users

def _dummify(df, column_names):
  dummies = pd.get_dummies(df, columns = column_names)
  return dummies
