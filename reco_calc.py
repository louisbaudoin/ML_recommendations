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

# Import user_characteristics_calc
import users_characteristics_calc


def get_metrics_batch(test_size, number_of_users_predicted, data_ratings, top_actual, top_predicted, data_users, weight_ratings):
  # Instantiate dataframe output
  output = pd.DataFrame(columns = ['number_within', 'mean_error', 'user_id'])
  # get array of user ids to test
  users_array = _get_users_to_test(number_of_users_predicted, test_size, data_ratings)
  # Initialize counter
  count = 0
  # Iterate on users
  for user_id in users_array:
    results = get_metrics(user_id, test_size, data_ratings, top_actual, top_predicted, data_users, weight_ratings)
    output.loc[user_id, 'number_within'] = results['number_within']
    output.loc[user_id, 'mean_error'] = results['mean_error']
    output.loc[user_id, 'user_id'] = user_id
    count += 1
  return output

# returns metrics for a given user
def get_metrics(user_id, test_size, data_ratings, top_actual, top_predicted, data_users, weight_ratings):
  # call get_ratings and get predicted and actual ratings
  ratings = get_ratings(user_id, test_size, data_ratings, data_users, weight_ratings)
  predicted_cut = ratings['cut_predicted_ratings']
  actual_cut = ratings['cut_actual_ratings']
  all_predicted = ratings['all_predicted_ratings']
  all_actual = ratings['all_actual_ratings']
  # call methods to calculate metrics
  number_of_movies_within_rated = _number_of_movies_within_rated(all_actual, all_predicted, top_actual, top_predicted)
  mean_error = _mean_error(actual_cut, predicted_cut)
  return {'mean_error': mean_error, 'number_within': number_of_movies_within_rated}

# Method to get the ratings without taking into account users characteristics and movies characteristics
def get_ratings(user_id, test_size, data_ratings, data_users, weight_ratings):
  # last 10 ratings of a given user + set movies_id as index
  data_ratings_cut = data_ratings[data_ratings['user_id'] == user_id].sort_values('timestamp', inplace = False, ascending = False).iloc[:test_size,:]
  data_ratings_cut.set_index('movies_id', drop = True, inplace = True)
  # list of last 10 indexes
  indexes_to_assess = data_ratings_cut.index.values
  #define bimatrix
  df_bimat = _bimatrix(user_id, indexes_to_assess, data_ratings)
  # Calculate correlations (here we can introduce influence of similarities with other users)
  corr_rating = df_bimat.corrwith(df_bimat[user_id], axis=0, drop=False)
  corr_user = compute_correlations_users(user_id, data_users)
  corr_weighted = _recompute_correlation(corr_rating, corr_user, weight_ratings)
  # Apply method (multiply line by line the correlations and the ratings)
  user_weights = df_bimat.apply(lambda row:_multiplication_line_by_line(row,corr_weighted), axis =1)
  # Create dummy
  dummy_df = user_weights.applymap(lambda x: 1 if pd.notnull(x) else 0)
  # get numerator (sum of correlations * ratings)
  user_numerator = user_weights.sum(axis = 1)
  # get denominator (sum of relevant correlations)
  user_denominator = dummy_df.apply(lambda row: _multiplication_line_by_line(row, corr_rating), axis = 1).sum(axis = 1)
  # get ratings
  user_ratings = user_numerator.div(user_denominator)
  # clean ratings : return 5 if >5 and 0 if negative
  user_ratings = user_ratings.map(_clean_user_ratings)
  # sort the best 'top'
  user_ratings.sort_values(ascending = False, inplace = True)
  # only the predictions for the ones that we assess
  cut_predicted_ratings = user_ratings[indexes_to_assess]
  # all_actual_ratings
  ordered_actual_ratings = data_ratings[data_ratings['user_id'] == user_id].set_index('movies_id', drop = True, inplace = False)
  ordered_actual_ratings.sort_values('rating', ascending = False, inplace = True)
  ordered_actual_ratings = ordered_actual_ratings['rating']
  return {'all_predicted_ratings': user_ratings, 'cut_predicted_ratings': cut_predicted_ratings,'cut_actual_ratings': data_ratings_cut['rating'], 'all_actual_ratings': ordered_actual_ratings}

# ----------- COMPUTE ALTERNATIVE CORRELATION
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

# ------------ OTHER PRIVATE METHODS
# Error on score ot those we kept as a test
def _mean_error(actual, predicted):
  tot = 0
  n = actual.shape[0]
  for i in actual.index.values:
    tot = tot + (actual[i] - predicted[i])**2
  return sqrt(tot)/n

# Return bimatrix
def _bimatrix(user_id, indexes_to_assess, data_ratings):
  # take out the ratings from data_ratings
  data_ratings_user = data_ratings.drop(data_ratings.index.isin(indexes_to_assess), inplace = False)
  # Create the bimatrix - movies X users
  df_bimat = pd.pivot_table(data_ratings_user, index = 'movies_id', columns = 'user_id', values = 'rating' )
  return df_bimat

# Method that multiplies the correlation vector with each line
def _multiplication_line_by_line(row, corr_user):
    return row.multiply(corr_user)

# clean ratings : return 5 if >5 and 0 if negative
def _clean_user_ratings(el):
  if el > 5:
    return 5
  elif el < 0:
    return 0
  else:
    return el

# Define metrics to assess performance
def _actual_ratings(user_id, number_of_users_predicted, test_size):
  # last 10 ratings of a given user
  return data_ratings[data_ratings['user_id'] == user_id].sort_values('timestamp', inplace = False, ascending = False).iloc[:test_size,:]

#select user_ids of those who have left at least min_ratings ratings --> we will test and iterate on all of these users
def _get_users_to_test(number_of_users_predicted, test_size, data_ratings):
  piv = pd.pivot_table(data_ratings, index ='user_id', values = 'rating', aggfunc = 'count')
  rand = [random.randint(0,piv.shape[0]) for r in xrange(number_of_users_predicted)]
  piv = piv[piv > test_size]
  piv = piv[rand]
  return piv.index.values


# % that are in our top 20 and that the user rated
def _number_of_movies_within_rated(all_actual, all_predicted, top_actual, top_predicted):
  # Instantiate metric
  tot = 0
  # Get only top_actual of all actual ratings
  actual_top = all_actual[:top_actual]
  # Get only top_predicted of all predicted ratings
  predicted_top = all_predicted[:top_predicted]
  # Iterate and increment metric
  for movie_id in actual_top.index.values:
      if movie_id in predicted_top.index.values:
          tot +=1
  # return the % of top_actual of actual movies that are in the first top_predicted
  return tot/actual_top.shape[0]

def _recompute_correlation(corr_rating, corr_users, weight_rating):
  weighted_corr_rating = corr_rating*weight_rating
  weighted_corr_users = corr_users*(1-weight_rating)
  return weighted_corr_rating.add(weighted_corr_users)




# def make_recommendations(user_ids, top):
#   # instantiate the builder for the df_results
#   hash_results = {}
#   for user_id in user_ids:
#     # build bimatrix
#     df_bimat = bimatrix(user_id, top)
#     # get ratings
#     user_ratings = get_ratings(user_id, top)
#     # store results in bui

# # Distance Euclidienne entre deux utilisateurs
# def sim_eucl(user_1, user_2):
#     # 0. Initialize output
#     tot_dist = 0
#     # 1. Returns list of films watched by user_1 and user_2
#     movies_user_1 = data_ratings[data_ratings['user_id'] == user_1]['movies_id']
#     movies_user_2 = data_ratings[data_ratings['user_id'] == user_2]['movies_id']

#     # 2. Create list of movies in common (if 0, return 0)
#     common_array = list(set(movies_user_1) & set(movies_user_2))
#     n = len(common_array)
#     if n == 0:
#         return 0
#     else:
#         # 3. Iterate on the list of common items
#         for movie in common_array:
#             first_coord = data_ratings[(data_ratings['user_id'] == user_1) & (data_ratings['movies_id'] == movie)]['rating'].values[0]
#             second_coord = data_ratings[(data_ratings['user_id'] == user_2) & (data_ratings['movies_id'] == movie)]['rating'].values[0]
#             common[movie] = [first_coord, second_coord]
#             tot_dist = tot_dist + (second_coord - first_coord)**2
#             print tot_dist
#         # 4. Return output - Let normalize the distance to make the distances between users comparable (bias if many vs few ratings in common)
#         return sqrt(tot_dist)/n
