# Dependencies
import pandas as pd
import csv
import numpy as np
import datetime
#from datetime import datetime
import calendar #necessary to convert a timestamp into a date
#from datetime import datetime
import time

# ----- PROCESS MOVIES
categories = ['Action', 'Adventure', 'Animation', "Children's", "Comedy", 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
def process_movies(path_movies):
  # Open and relabel
  data_movies = pd.read_csv(path_movies, sep = '::', error_bad_lines= True, engine = 'python', header = None)
  data_movies.columns = ['number', 'movie', 'categories']
  # Turn categories into a serie of dummies
  data_movies['categories'] = data_movies['categories'].map(_dummify)
  # expend the serie within the dataframe
  data_movies = data_movies.apply(lambda row: row.append(row['categories']), axis = 1)
  # delete the series
  del data_movies['categories']
  #relabel
  data_movies.columns = np.append(['movies_id', 'movie_name'], categories)
  return data_movies

# Turn categories into a serie
def _dummify(el):
    # Declare output
    output = np.zeros(18)
    # create an array of 3 (max) categories
    el = el.split('|')
    # Iterate on the categories
    for cat in el:
        if cat in categories:
            ind = categories.index(cat)
            output[ind] = int(1)
    return pd.Series(output)


# ----- PROCESS RATINGS
def process_ratings(path_ratings):
  # read
  data_ratings = pd.read_csv(path_ratings, sep = '::', error_bad_lines= True, engine = 'python', header = None)
  # relabel
  labels_ratings = ['user_id', 'movies_id', 'rating', 'timestamp']
  data_ratings.columns = labels_ratings
  #process timestamps
  data_ratings['timestamp'] = data_ratings['timestamp'].map(_to_datetime)
  return data_ratings

def _to_datetime(timestamp):
  date = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
  date_output = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
  return date_output


# ---- PROCESS USERS INFO
def process_users(path_users):
  # read
  data_users = pd.read_csv(path_users, sep = '::', error_bad_lines= True, engine = 'python', header = None)
  # relabel
  users_labels = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
  data_users.columns = users_labels
  # convert age
  data_users['age'] = data_users['age'].map(convert_age)
  # convert occupation
  data_users['occupation'] = data_users['occupation'].map(convert_occupation)
  return data_users


# age dict
convert_age = {1:  "Under 18",18:  "18-24",25:  "25-34", 35:  "35-44", 45:  "45-49", 50:  "50-55", 56:  "56+"}
# occupation dict
convert_occupation = {0:  "other",
                    1:  "academic/educator",
                    2:  "artist",
                    3:  "clerical/admin",
                    4:  "college/grad student",
                    5:  "customer service",
                    6:  "doctor/health care",
                    7:  "executive/managerial",
                    8:  "farmer",
                    9:  "homemaker",
                    10:  "K-12 student",
                    11:  "lawyer",
                    12:  "programmer",
                    13:  "retired",
                    14:  "sales/marketing",
                    15:  "scientist",
                    16:  "self-employed",
                    17:  "technician/engineer",
                    18:  "tradesman/craftsman",
                    19:  "unemployed",
                    20:  "writer"
                  }








