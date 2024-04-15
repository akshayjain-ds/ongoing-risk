# Databricks notebook source
# MAGIC %run ./set_up

# COMMAND ----------

import gc
import pendulum
import calendar
import random
from calendar import monthrange
from pyspark.sql import DataFrame
import functools
from pyspark.sql.functions import to_timestamp
from typing import List
from pyspark.sql import DataFrame
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
from typing import List
gc.enable()

def get_random_day_of_month(year: int, month: int) -> datetime.date:
    """Returns a deterministic/repeatable random day of the month to include in the training dataset

    Args:
      year (int): Calendar Year
      month (int): Calendar Month

    Returns:
      datetime.date: Date object representing the random day of the month
    """
    random.seed(datetime.date(year=year, month=month, day=1).strftime(
        '%Y-%m-%d'))  # To ensure same random dates are picked for historical training
    dates = calendar.Calendar().itermonthdates(year, month)  # Iterator defined for days of the month
    return random.choice([date for date in dates if date.month == month])  # Random choice of the day in month


def get_first_day_of_month(year: int, month: int) -> datetime.date:
    """Returns a deterministic/repeatable random day of the month to include in the training dataset

    Args:
      year (int): Calendar Year
      month (int): Calendar Month

    Returns:
      datetime.date: Date object representing the first day of the month
    """
    random.seed(datetime.date(year=year, month=month, day=1).strftime(
        '%Y-%m-%d'))  # To ensure same random dates are picked for historical training
    dates = calendar.Calendar().itermonthdates(year, month)  # Iterator defined for days of the month
    return [date for date in dates if date.month == month][0]  # First day in month


def get_multiple_random_day_of_month(year: int, month: int) -> datetime.date:
    """Returns a deterministic/repeatable random day of the month to include in the training dataset

    Args:
      year (int): Calendar Year
      month (int): Calendar Month

    Returns:
      datetime.date: Date Objects representing the 7 random days of the month
    """
    random.seed(datetime.date(year=year, month=month, day=1).strftime(
        '%Y-%m-%d'))  # To ensure same random dates are picked for historical training
    dates = calendar.Calendar().itermonthdates(year, month)  # Iterator defined for days of the month
    return random.choices([date for date in dates if date.month == month], k=3)  # Random choice of the day in month


def get_randomized_dates(start_year: int, start_month: int, end_year: int, end_month: int, multiple_day = 'False') -> \
List[str]:
    """Returns a deterministic set of randomized dates from each month in the time period specified

    Args:
      start_year (int): Timeperiod starting year
      start_month (int): Timeperiod starting month
      end_year (int): Timeperiod ending year
      end_month (int): Timeperiod ending month

    Returns:
      List[str]: List of dates from each month in the time period specified
    """
    start_date = pendulum.date(start_year, start_month, 1)
    end_date = pendulum.date(end_year, end_month, monthrange(end_year, end_month)[1])

    if multiple_day == 'False':
        print('Using single day')
        randomized_dates = []
        for each_month in pendulum.period(start_date, end_date).range(
                'months'):  # Iterate over each month in time period
            randomized_dates.append(get_random_day_of_month(each_month.year, each_month.month).strftime('%Y-%m-%d'))
        return randomized_dates

    if multiple_day == 'First':
        print('Using first day')
        randomized_dates = []
        for each_month in pendulum.period(start_date, end_date).range(
                'months'):  # Iterate over each month in time period
            randomized_dates.append(get_first_day_of_month(each_month.year, each_month.month).strftime('%Y-%m-%d'))
        return randomized_dates

    else:
        print('Using Multiple days')
        randomized_dates = []
        for each_month in pendulum.period(start_date, end_date).range(
                'months'):  # Iterate over each month in time period
            selected_dates = get_multiple_random_day_of_month(each_month.year, each_month.month)
            for chosen_date in selected_dates:
                randomized_dates.append(chosen_date.strftime('%Y-%m-%d'))
        return randomized_dates
      
def setup_training_dataset_spine(start_year: int, start_month: int, end_year: int, end_month: int) -> DataFrame:
  """Returns a deterministic set of randomized dates from each month in the time period specified
  
  Args:
    start_year (int): Timeperiod starting year
    start_month (int): Timeperiod starting month
    end_year (int): Timeperiod ending year
    end_month (int): Timeperiod ending month
    
  Returns:
    pyspark.sql.DataFrame: List of active companies to be included in the spine for the entire training period
  """
  spine_df = None
  if start_year == end_year and start_month == end_month: # if spine requested for only one month
    feature_date = get_random_day_of_month(start_year, start_month).strftime('%Y-%m-%d')  # Choose random day of month
    spine_df = setup_active_companies_spine(feature_date)   # get active companies to include in spine
  else:  # if spine requested for a time period i.e. multiple months
    feature_dates = get_randomized_dates(start_year, start_month, end_year, end_month)  # Get random dates for entire time period
    spines_list = [setup_active_companies_spine(each_date) for each_date in feature_dates]  # Get active companies on each of the date
    spine_df = functools.reduce(DataFrame.union, spines_list)   # Union all spark dataframes to make consolidate spine
  return spine_df

# COMMAND ----------

# MAGIC %run ./train_metadata

# COMMAND ----------

# MAGIC %run ./queries

# COMMAND ----------

SCOPE = "tecton"
SNOWFLAKE_DATABASE = "TIDE"
SNOWFLAKE_USER = dbutils.secrets.get(scope=SCOPE, key="SNOWFLAKE_DATABRICKS_USER")
SNOWFLAKE_PASSWORD = dbutils.secrets.get(scope=SCOPE, key="SNOWFLAKE_DATABRICKS_PASSWORD")
SNOWFLAKE_ACCOUNT = dbutils.secrets.get(scope=SCOPE, key="SNOWFLAKE_DATABRICKS_ACCOUNT")
SNOWFLAKE_WAREHOUSE = "DATABRICKS_WH"
SNOWFLAKE_SCHEMA = "DB_TIDE"
SNOWFLAKE_ROLE = "DATABRICKS_ROLE"

# snowflake connection options
CONNECTION_OPTIONS = dict(sfUrl=SNOWFLAKE_ACCOUNT,
                          sfUser=SNOWFLAKE_USER,
                          sfPassword=SNOWFLAKE_PASSWORD,
                          sfDatabase=SNOWFLAKE_DATABASE,
                          sfSchema=SNOWFLAKE_SCHEMA,
                          sfWarehouse=SNOWFLAKE_WAREHOUSE,
                          sfRole=SNOWFLAKE_ROLE)


def spark_connector(query_string: str) -> DataFrame:
    """Returns spark dataframe as a result set of the input query string

    Args:
      query_string (str): Query that needs to be run using spark

    Returns:
      pyspark.sql.DataFrame: Resulting rows for the query
    """
    df = spark.read \
        .format("snowflake") \
        .options(**CONNECTION_OPTIONS) \
        .option("query", query_string) \
        .load().cache()
    return df

# COMMAND ----------

import os
import pyspark.sql.functions as F 
from pyspark.sql.functions import col
import numpy as np
from pyspark.sql.types import StringType

def setup_active_companies_spine(query: str, feature_date: str, ntt_period: int) -> DataFrame:
  """Returns spark dataframe with spine representing companies for ongoing risk
  
  Args:
    feature_date (str): Date on which features are calculated
    feature_date (str): maximum days on books (NtT definition)
    
  Returns:
    pyspark.sql.DataFrame: List of active companies as on feature date
  """
  spine_df = spark_connector(query.format(feature_date=feature_date,
                                          ntt_period=ntt_period))
  # spine_df = spine_df.withColumnRenamed("COMPANY_ID","company_id") # Rename company-id column
  # spine_df = spine_df.withColumn("company_id",col("company_id").cast(StringType()))
  # spine_df = spine_df.withColumn("timestamp", to_timestamp("timestamp")) # Casting timestamp column
  return spine_df

# COMMAND ----------

def setup_training_dataset_spine(query:str, start_year: int, start_month: int, end_year: int, end_month: int, ntt_period: int) -> DataFrame:
  """Returns a deterministic set of randomized dates from each month in the time period specified
  
  Args:
    start_year (int): Timeperiod starting year
    start_month (int): Timeperiod starting month
    end_year (int): Timeperiod ending year
    end_month (int): Timeperiod ending month
    
  Returns:
    pyspark.sql.DataFrame: List of active companies to be included in the spine for the entire training period
  """
  spine_df = None
  if start_year == end_year and start_month == end_month: # if spine requested for only one month
    feature_date = get_random_day_of_month(start_year, start_month).strftime('%Y-%m-%d')  # Choose random day of month
    spine_df = setup_active_companies_spine(query, feature_date, ntt_period)   # get active companies to include in spine
  else:  # if spine requested for a time period i.e. multiple months
    feature_dates = get_randomized_dates(start_year, start_month, end_year, end_month, multiple_day = 'False' if day_of_month == 'random_day' else 'First')  # Get random dates for entire time period
    spines_list = [setup_active_companies_spine(query, feature_date, ntt_period) for feature_date in feature_dates]  # Get active companies on each of the date
    spine_df = functools.reduce(DataFrame.union, spines_list)   # Union all spark dataframes to make consolidate spine
  return spine_df

# COMMAND ----------

dataset_spine = setup_training_dataset_spine(
  query = ntt_ftt_query if include_ftt else ntt_query,
  start_year=pd.to_datetime(timestamp_start_date).year,  
  start_month=pd.to_datetime(timestamp_start_date).month,
  end_year=pd.to_datetime(scoring_date).year, 
  end_month=pd.to_datetime(scoring_date).month,
  ntt_period=ntt_period)

dataset_df = dataset_spine.toPandas()
dataset_df.columns= dataset_df.columns.str.lower()
dataset_df.rename(columns={'"timestamp"':"timestamp"}, inplace=True)
dataset_df.shape

# COMMAND ----------

dataset_df.head()

# COMMAND ----------

dataset_df = dataset_df[~pd.isnull(pd.to_numeric(dataset_df[id1], errors='coerce'))]
dataset_df = dataset_df.assign(

  company_created_on = lambda col: pd.to_datetime(col[created_on]),
  timestamp = lambda col: pd.to_datetime(col[timestamp]),
  approved_at = lambda col:  pd.to_datetime(col['approved_at']),

  days_on_books = lambda col:  pd.to_numeric(col['days_on_books'], errors='coerce'),
  is_ftt = lambda col:  pd.to_numeric(col['is_ftt'], errors='coerce'),
  days_to_transact = lambda col:  pd.to_numeric(col['days_to_transact'], errors='coerce'),
  days_remaining_as_ntt_ftt = lambda col:  pd.to_numeric(col['days_remaining_as_ntt_ftt'], errors='coerce'),

  is_ntt = lambda col: np.where(col['days_on_books'] <= ntt_period, 1, 0),      
)
dataset_df.shape

# COMMAND ----------

assert dataset_df[[id1, timestamp]].shape == dataset_df[[id1, timestamp]].drop_duplicates().shape

# COMMAND ----------

dataset_df[timestamp].min(), dataset_df[timestamp].max()

# COMMAND ----------

appf = spark_connector(appf_query.format(from_date=timestamp_start_date, 
                                         to_date=str(pd.to_datetime(scoring_date).date() + timedelta(days=ntt_period))))
appf = appf.toPandas()
appf.columns = [col.split("__")[1].lower() if col.__contains__("__") else col.lower() for col in appf.columns]
appf.drop_duplicates(subset = [id1, 'transaction_at', 'app_fraud_type', 'amount'], inplace=True)
appf.shape

# COMMAND ----------

duplicates = dataset_df.merge(
  appf[[id1, 'transaction_at', 'app_fraud_type', 'amount']], 
  on=[id1], how='left')
duplicates.drop_duplicates(subset=[id1, timestamp, 'transaction_at'], inplace=True)
duplicates.shape

# COMMAND ----------

duplicates = duplicates.assign(

  transaction_at = lambda col: pd.to_datetime(col['transaction_at']),

  amount = lambda col: pd.to_numeric(col['amount'], errors='coerce'), 

  days_to_fraud = lambda col: col[['transaction_at', timestamp]].apply(lambda row: 
    (row['transaction_at'] - row[timestamp]).days if row['transaction_at'] > row[timestamp] else np.NaN,
    axis=1),
  
  is_app_fraud_as_ntt_ftt = lambda col: np.where(
    col['days_to_fraud'] <= col['days_remaining_as_ntt_ftt'], True, False),
  
  days_to_fraud_as_ntt_ftt = lambda col: np.where(
    col['days_to_fraud'] <= col['days_remaining_as_ntt_ftt'], col['days_to_fraud'], np.NaN),
  
  app_fraud_amount_as_ntt_ftt = lambda col: np.where(
    col['days_to_fraud'] <= col['days_remaining_as_ntt_ftt'], col['amount'], np.NaN),
  
  app_fraud_type_as_ntt_ftt = lambda col: np.where(
    col['days_to_fraud'] <= col['days_remaining_as_ntt_ftt'], col['app_fraud_type'], None)
)

duplicates.drop(columns=['transaction_at', 'app_fraud_type', 'amount', 'days_to_fraud'], 
                inplace=True)
duplicates.drop_duplicates(subset=[id1, timestamp, 
                                   'is_app_fraud_as_ntt_ftt', 'days_to_fraud_as_ntt_ftt', 'app_fraud_amount_as_ntt_ftt', 'app_fraud_type_as_ntt_ftt'],
                           inplace=True)
duplicates.sort_values(by=[id1, timestamp, 
                          'is_app_fraud_as_ntt_ftt', 'days_to_fraud_as_ntt_ftt', 'app_fraud_amount_as_ntt_ftt', 'app_fraud_type_as_ntt_ftt'], 
                       inplace=True)
duplicates.shape

# COMMAND ----------

is_app_fraud_as_ntt_ftt_agg = duplicates.groupby([id1, timestamp],
                                             as_index = False)[['is_app_fraud_as_ntt_ftt']].agg({'is_app_fraud_as_ntt_ftt': lambda x: list(x)})

days_to_fraud_as_ntt_ftt_agg = duplicates.groupby([id1, timestamp],
                                                  as_index = False)[['days_to_fraud_as_ntt_ftt']].agg({'days_to_fraud_as_ntt_ftt': lambda x: list(x)})
days_to_fraud_as_ntt_ftt_agg['days_to_fraud_as_ntt_ftt'] = days_to_fraud_as_ntt_ftt_agg['days_to_fraud_as_ntt_ftt'].apply(lambda x: list(filter(lambda v: v is not [np.NaN, None], x)))

app_fraud_amount_as_ntt_ftt_agg = duplicates.groupby([id1, timestamp], 
                                                    as_index = False)[['app_fraud_amount_as_ntt_ftt']].agg({'app_fraud_amount_as_ntt_ftt': lambda x: list(x)})
app_fraud_amount_as_ntt_ftt_agg['app_fraud_amount_as_ntt_ftt'] = app_fraud_amount_as_ntt_ftt_agg['app_fraud_amount_as_ntt_ftt'].apply(lambda x: list(filter(lambda v: v is not [np.NaN, None], x)))

app_fraud_type_as_ntt_ftt_agg = duplicates.groupby([id1, timestamp], 
                                                  as_index = False)[['app_fraud_type_as_ntt_ftt']].agg({'app_fraud_type_as_ntt_ftt': lambda x: list(x)})
app_fraud_type_as_ntt_ftt_agg['app_fraud_type_as_ntt_ftt'] = app_fraud_type_as_ntt_ftt_agg['app_fraud_type_as_ntt_ftt'].apply(lambda x: list(filter(lambda v: v is not [np.NaN, None], x)))

is_app_fraud_as_ntt_ftt_agg.shape, days_to_fraud_as_ntt_ftt_agg.shape, app_fraud_amount_as_ntt_ftt_agg.shape, app_fraud_type_as_ntt_ftt_agg.shape

# COMMAND ----------

del duplicates
gc.collect()

# COMMAND ----------

dataset_df = dataset_df.merge(
  is_app_fraud_as_ntt_ftt_agg, on=[id1, timestamp]).merge(
    days_to_fraud_as_ntt_ftt_agg, on=[id1, timestamp]).merge(
      app_fraud_amount_as_ntt_ftt_agg, on=[id1, timestamp]).merge(
        app_fraud_type_as_ntt_ftt_agg, on=[id1, timestamp])

dataset_df.shape

# COMMAND ----------

dataset_df = dataset_df.assign(

  is_app_fraud_30d = lambda col: col['days_to_fraud_as_ntt_ftt']
  .apply(lambda x: any(np.where(np.array(x, dtype=float)<=30, True, False))),
  is_app_fraud_45d = lambda col: col['days_to_fraud_as_ntt_ftt']
  .apply(lambda x: any(np.where(np.array(x, dtype=float)<=45, True, False))),
  is_app_fraud_60d = lambda col: col['days_to_fraud_as_ntt_ftt']
  .apply(lambda x: any(np.where(np.array(x, dtype=float)<=60, True, False))),
  is_app_fraud_90d = lambda col: col['days_to_fraud_as_ntt_ftt']
  .apply(lambda x: any(np.where(np.array(x, dtype=float)<=90, True, False))),

  days_to_fraud_30d = lambda col: col['days_to_fraud_as_ntt_ftt']
  .apply(lambda x: np.array(x, dtype=float).min() if np.array(x, dtype=float).min()<=30 else np.NaN),
  days_to_fraud_45d = lambda col: col['days_to_fraud_as_ntt_ftt']
  .apply(lambda x: np.array(x, dtype=float).min() if np.array(x, dtype=float).min()<=45 else np.NaN),
  days_to_fraud_60d = lambda col: col['days_to_fraud_as_ntt_ftt']
  .apply(lambda x: np.array(x, dtype=float).min() if np.array(x, dtype=float).min()<=60 else np.NaN),
  days_to_fraud_90d = lambda col: col['days_to_fraud_as_ntt_ftt']
  .apply(lambda x: np.array(x, dtype=float).min() if np.array(x, dtype=float).min()<=90 else np.NaN),

  app_fraud_amount_30d = lambda col: col[['days_to_fraud_as_ntt_ftt', 'app_fraud_amount_as_ntt_ftt']]
  .apply(lambda row: np.dot(
    np.where(np.array(row['days_to_fraud_as_ntt_ftt'], dtype=float)<=30, 1, 0), 
    np.array(row['app_fraud_amount_as_ntt_ftt'], dtype=float)
    ), axis=1),
  app_fraud_amount_45d = lambda col: col[['days_to_fraud_as_ntt_ftt', 'app_fraud_amount_as_ntt_ftt']]
  .apply(lambda row: np.dot(
    np.where(np.array(row['days_to_fraud_as_ntt_ftt'], dtype=float)<=45, 1, 0), 
    np.array(row['app_fraud_amount_as_ntt_ftt'], dtype=float)
    ), axis=1),
  app_fraud_amount_60d = lambda col: col[['days_to_fraud_as_ntt_ftt', 'app_fraud_amount_as_ntt_ftt']]
  .apply(lambda row: np.dot(
    np.where(np.array(row['days_to_fraud_as_ntt_ftt'], dtype=float)<=60, 1, 0), 
    np.array(row['app_fraud_amount_as_ntt_ftt'], dtype=float)
    ), axis=1),
  app_fraud_amount_90d = lambda col: col[['days_to_fraud_as_ntt_ftt', 'app_fraud_amount_as_ntt_ftt']]
  .apply(lambda row: np.dot(
    np.where(np.array(row['days_to_fraud_as_ntt_ftt'], dtype=float)<=90, 1, 0), 
    np.array(row['app_fraud_amount_as_ntt_ftt'], dtype=float)
    ), axis=1),
  
  app_fraud_type_30d = lambda col: col[['days_to_fraud_as_ntt_ftt', 'app_fraud_type_as_ntt_ftt']]
  .apply(lambda row: [y for x, y in zip(row['days_to_fraud_as_ntt_ftt'], 
                                        row['app_fraud_type_as_ntt_ftt']) if float(x)<=30]
         , axis=1),
  app_fraud_type_45d = lambda col: col[['days_to_fraud_as_ntt_ftt', 'app_fraud_type_as_ntt_ftt']]
  .apply(lambda row: [y for x, y in zip(row['days_to_fraud_as_ntt_ftt'], 
                                        row['app_fraud_type_as_ntt_ftt']) if float(x)<=45]
         , axis=1),
  app_fraud_type_60d = lambda col: col[['days_to_fraud_as_ntt_ftt', 'app_fraud_type_as_ntt_ftt']]
  .apply(lambda row: [y for x, y in zip(row['days_to_fraud_as_ntt_ftt'], 
                                        row['app_fraud_type_as_ntt_ftt']) if float(x)<=60]
         , axis=1),
  app_fraud_type_90d = lambda col: col[['days_to_fraud_as_ntt_ftt', 'app_fraud_type_as_ntt_ftt']]
  .apply(lambda row: [y for x, y in zip(row['days_to_fraud_as_ntt_ftt'], 
                                        row['app_fraud_type_as_ntt_ftt']) if float(x)<=90]
         , axis=1)
)
dataset_df.shape

# COMMAND ----------

dataset_df['is_app_fraud_30d'].mean(), dataset_df['is_app_fraud_45d'].mean(), dataset_df['is_app_fraud_60d'].mean(), dataset_df['is_app_fraud_90d'].mean()

# COMMAND ----------

dataset_df['days_to_fraud_30d'].mean(), dataset_df['days_to_fraud_45d'].mean(), dataset_df['days_to_fraud_60d'].mean(), dataset_df['days_to_fraud_90d'].mean()

# COMMAND ----------

dataset_df[dataset_df['is_app_fraud_30d']][id1].nunique(), dataset_df[dataset_df['is_app_fraud_45d']][id1].nunique(), dataset_df[dataset_df['is_app_fraud_60d']][id1].nunique(), dataset_df[dataset_df['is_app_fraud_90d']][id1].nunique()

# COMMAND ----------

dataset_df['app_fraud_amount_30d'].mean(), dataset_df['app_fraud_amount_45d'].mean(), dataset_df['app_fraud_amount_60d'].mean(), dataset_df['app_fraud_amount_90d'].mean()

# COMMAND ----------

dataset_df.head()

# COMMAND ----------

training_df = dataset_df[(dataset_df[timestamp] >= pd.to_datetime(timestamp_start_date)) & 
                        (dataset_df[timestamp] <= pd.to_datetime(timestamp_end_date))]
                        
scoring_df = dataset_df[~((dataset_df[timestamp] >= pd.to_datetime(timestamp_start_date)) & 
                        (dataset_df[timestamp] <= pd.to_datetime(timestamp_end_date)))]

training_df.shape, scoring_df.shape

# COMMAND ----------

company_counts = training_df.groupby('company_id').size()
training_df['weight'] = training_df['company_id'].apply(lambda x: 1.0 / company_counts[x])
print(np.around(company_counts.mean(), 0), np.around(company_counts.std(), 0))

# COMMAND ----------

company_counts.value_counts().sort_index()

# COMMAND ----------

company_counts = scoring_df.groupby('company_id').size()
scoring_df['weight'] = scoring_df['company_id'].apply(lambda x: 1.0 / company_counts[x])

# COMMAND ----------

numeric_colums = [col for col in training_df.columns if col not in [id1, id2, timestamp, created_on, 'approved_at', 'rejected_at'] + [f for f in training_df.columns if col.__contains__("app_fraud_type")]]
numeric_colums

# COMMAND ----------

for f in numeric_colums:
  training_df[f] = pd.to_numeric(training_df[f], errors='coerce')
  scoring_df[f] = pd.to_numeric(scoring_df[f], errors='coerce')

# COMMAND ----------

pd.isnull(training_df).sum()/training_df.shape[0]

# COMMAND ----------

for col in ['days_on_books', 'days_to_transact', 'is_registered', 'is_ftt', 
            f"is_app_fraud_as_{'ntt_ftt' if include_ftt else 'ntt'}_30d", f"days_to_fraud_as_{'ntt_ftt' if include_ftt else 'ntt'}_30d",
            f"is_app_fraud_as_{'ntt_ftt' if include_ftt else 'ntt'}_60d", f"days_to_fraud_as_{'ntt_ftt' if include_ftt else 'ntt'}_60d", 
            f"is_app_fraud_as_{'ntt_ftt' if include_ftt else 'ntt'}_90d", f"days_to_fraud_as_{'ntt_ftt' if include_ftt else 'ntt'}_90d"]:
  
  missing_index = pd.isnull(training_df[col].astype(float))
  print("unweighted", col, 
        np.average(training_df[~missing_index][col].astype(float)))
  print("weighted", col, 
        np.average(training_df[~missing_index][col].astype(float),
                    weights=training_df[~missing_index]['weight']))

# COMMAND ----------

training_df.groupby(["timestamp"])[['days_on_books']].mean().plot()

# COMMAND ----------

scoring_df.head()

# COMMAND ----------

pd.isnull(scoring_df).sum()

# COMMAND ----------

for col in ['days_on_books', 'days_to_transact', 'is_registered', 'is_ftt', 
            f"is_app_fraud_as_{'ntt_ftt' if include_ftt else 'ntt'}_30d", f"days_to_fraud_as_{'ntt_ftt' if include_ftt else 'ntt'}_30d",
            f"is_app_fraud_as_{'ntt_ftt' if include_ftt else 'ntt'}_60d", f"days_to_fraud_as_{'ntt_ftt' if include_ftt else 'ntt'}_60d", 
            f"is_app_fraud_as_{'ntt_ftt' if include_ftt else 'ntt'}_90d", f"days_to_fraud_as_{'ntt_ftt' if include_ftt else 'ntt'}_90d"]:
  
  missing_index = pd.isnull(scoring_df[col].astype(float))
  print(col, 
        np.average(scoring_df[~missing_index][col].astype(float)))

# COMMAND ----------

dataset_df = pd.concat([training_df, scoring_df])
dataset_df.drop(columns=['weight'], inplace=True)
dataset_df.rename(columns={'"timestamp"': "timestamp"}, inplace=True)
dataset_df = dataset_df.drop_duplicates()
dataset_df.shape

# COMMAND ----------

dataset_df = training_df.assign(

  company_created_on = lambda col: pd.to_datetime(col[created_on]),
  timestamp = lambda col: pd.to_datetime(col[timestamp]),
  approved_at = lambda col:  pd.to_datetime(col['approved_at'])        
)
dataset_df['is_ntt'].mean(), training_df['is_ftt'].mean()

# COMMAND ----------

dataset_df[dataset_df[target_b]==1][id1].nunique(), dataset_df[dataset_df[target_b]==1][target_c].sum()

# COMMAND ----------

dataset_df.groupby(timestamp)[[id1]].count()

# COMMAND ----------

dataset_df["app_fraud_type_as_ntt_ftt_30d"].unique()

# COMMAND ----------

dataset_df.groupby(timestamp)[['days_on_books', f"days_remaining_as_{'ntt_ftt' if include_ftt else 'ntt'}"]].mean()

# COMMAND ----------

dataset_df[dataset_df[target_b]==1].groupby(timestamp)[[target_d, target_c]].mean()

# COMMAND ----------

dataset_df.shape

# COMMAND ----------

dataset_df.to_csv(f"/Workspace/Shared/Decisioning/Strawberries/ongoing-ntt_mp_risk-v1/artefacts/{'ntt_ftt' if include_ftt else 'ntt'}_{day_of_month}_base_{train_start_date}_{val_end_date}.csv.gz", index=False, compression='gzip')

# COMMAND ----------


