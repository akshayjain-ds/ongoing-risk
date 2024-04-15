# Databricks notebook source
# MAGIC %md 
# MAGIC
# MAGIC <h1>Ongoing Risk - Utils</h1>
# MAGIC
# MAGIC This notebook holds the definitions of the utils methods. This is imported in the training and evaluate & setup_training_dataset notebooks.

# COMMAND ----------

# Install dependencies
# %pip install --upgrade pip
# %pip install scikit-learn==1.1.0
# %pip install mlflow==1.17.0
# %pip install matplotlib==3.2
# %pip install feature-engine==1.5.0
# %pip install category_encoders==2.5.1.post0
# %pip install shap==0.35.0
# %pip install scikit-survival==0.18.0
# %pip install numpy==1.21.0
# %pip install tecton==0.6.10

# COMMAND ----------

import tecton
from tecton import conf
import logging
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
from typing import List
from sksurv.ensemble import RandomSurvivalForest
from sklearn.calibration import CalibratedClassifierCV
from sksurv.datasets import get_x_y
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
import mlflow
from sklearn.calibration import calibration_curve
import seaborn as sns
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
%matplotlib inline

# Model features

DROP_NA_COLS = ('is_registered', 'mkyc_ind', 'fincrime90_event', 'fincrime90_duration', 'age_at_completion')
IFRE_COLS = ('applicant_id_type', 'applicant_id_country_issue', 'applicant_postcode', 'section_description')
TRANSACTION_COLS = (
'avg_deposit', 'avg_withdrawal', 'max_deposit', 'max_withdrawal', 'pct_round_txns', 'pct_unique_txns', 'tester_pmt_cnt',
'cashtxns_latehrs',
'cardpmt_wtd_pct', 'fastpmt_wtd_pct', 'ddebit_wtd_pct', 'cardwtd_wtd_pct', 'outpmt_wtd_pct', 'fastpmt_dep_pct',
'cash_dep_pct', 'inpmt_dep_pct',
'fastpmt_beneficiaries', 'fastpmt_benefactors', 'cardpmt_acceptors', 'ddebit_beneficiaries',
'deposit_wtd_frequency_ratio', 'hmrc_txns_cnt',
'xero_txns_cnt', 'pos_atm_locations', 'card_pans_cnt', 'high_card_pmts', 'high_card_wtds', 'high_fpmt_out',
'high_pmt_out', 'high_ddebit',
'high_fpmt_in', 'max_cash_deposits', 'high_pmt_in')
IMPUTER_COLS = ('days_since_incorp', 'company_type',) + TRANSACTION_COLS
EVENT_COLS = ('fincrime90_event', 'fincrime90_duration')
EVENT_COL = 'fincrime90_event'
DURATION_COL = 'fincrime90_duration'
CATEGORICAL_COLS = ('is_registered', 'mkyc_ind', 'company_type')
FEATURES = ('days_since_incorp', 'is_registered', 'company_type', 'mkyc_ind',
            'age_at_completion') + IFRE_COLS + TRANSACTION_COLS
print(FEATURES)
NUMERICAL_COLS = tuple(x for x in FEATURES if x not in CATEGORICAL_COLS and x not in IFRE_COLS)

# Get all active companies with atleast-one cleared transaction as on FEATURE_DATE
# Query tecton features with the above company set and date

GET_ACTIVE_MEMBERS_QUERY = """
SELECT c.company_id as company_id, '{FEATURE_DATE}' as timestamp
FROM 	TIDE.DB_TIDE.TRANSACTION t JOIN TIDE.PRES_CORE.ACCOUNTS a ON t.accountid = a.account_id
			JOIN TIDE.PRES_CORE.COMPANIES c ON c.company_id = a.company_id JOIN TIDE.PRES_CORE.MEMBERSHIPS m ON c.member_id = m.member_id
WHERE (m.is_rejected = 0 OR (m.is_rejected = 1 and m.rejected_at_clean > '{FEATURE_DATE}')) AND 
			t.amount <> 0 and t.transactiontype NOT IN ('Fee', 'Unknown') and t.transactiondate < '{FEATURE_DATE}' and t.fullycleared = 1
GROUP BY 1
"""

# Get all active companies including unfunded accounts (i.e without a single cleared transaction) as on FEATURE_DATE
# Query tecton features with the above company set and date

GET_MEMBERS_QUERY = """
SELECT c.company_id as company_id, '{FEATURE_DATE}' as timestamp
FROM   TIDE.PRES_CORE.COMPANIES c JOIN TIDE.PRES_CORE.MEMBERSHIPS m ON c.member_id = m.member_id
WHERE  m.is_approved = 1 and m.approved_at_clean < '{FEATURE_DATE}' and (m.is_rejected = 0 OR (m.is_rejected = 1 and m.rejected_at_clean > '{FEATURE_DATE}'))
GROUP BY 1
"""

SAR_MEMBERS_QUERY = """ WITH SARS_RECORDS AS (
          SELECT  REGEXP_SUBSTR(TRIM(jira_tickets.ticket_summary), '[0-9]{{4,}}') AS company_id,
                      TO_DATE(DATE_TRUNC('DAY', MIN(jira_ticket_changes.change_at))) AS first_sar_created_dt
          FROM    TIDE.PRES_JIRA.JIRA_TICKETS AS jira_tickets LEFT JOIN TIDE.PRES_JIRA.JIRA_TICKET_CHANGES AS jira_ticket_changes
                          ON jira_tickets.TICKET_ID = jira_ticket_changes.TICKET_ID
          WHERE	jira_tickets.PROJECT_KEY = 'RCM' AND TRIM(jira_tickets.issue_type) IN ('TM alert', 'Risk case') AND
                       (jira_tickets.JIRA_TICKET_STATUS IS NULL OR jira_tickets.JIRA_TICKET_STATUS <> 'Duplicates') AND
                       (NOT (jira_tickets.is_subtask = 1 ) OR (jira_tickets.is_subtask = 1 ) IS NULL) AND
                       jira_ticket_changes.NEW_VALUE IN ('SAR', 'Tide Review', 'PPS Review', 'Submit to NCA', 'NCA Approval', 'NCA Refusal', 'Clear funds', 'Off-board customer')
          GROUP BY 1
          )
          SELECT A.company_id, A.first_sar_created_dt
          FROM SARS_RECORDS A JOIN TIDE.PRES_CORE.COMPANIES c ON A.company_id = c.company_id JOIN TIDE.PRES_CORE.MEMBERSHIPS m ON c.member_id = m.member_id
          WHERE m.is_approved = 1 AND A.first_sar_created_dt BETWEEN '{old_date}' AND '{new_date}' """

GET_PREDICTIONS_QUERY = """SELECT * FROM "TIDE"."ONGOING_RISK_SERVICE"."ONGOING_RISK_PREDICTIONS" WHERE EVALUATION_DATE BETWEEN '{old_date}' AND '{new_date}'"""

# Snowflake credentials

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


def setup_dataset_name(start_year: int, start_month: int, end_year: int, end_month: int) -> str:
    """Returns a datasetname for the time period specified

    Args:
      start_year (int): Timeperiod starting year
      start_month (int): Timeperiod starting month
      end_year (int): Timeperiod ending year
      end_month (int): Timeperiod ending month

    Returns:
      str: Name of the dataset
    """
    month_abbr = lambda month: datetime.datetime.strptime(str(month), "%m").strftime("%m")
    dataset_prefix = "ofre"
    return f"{dataset_prefix}_{month_abbr(start_month)}_{start_year}_{month_abbr(end_month)}_{end_year}"
    # if start_year == end_year and start_month == end_month:
    #  return f"{dataset_prefix}_{start_year}_{month_abbr(start_month)}"
    # else:
    #  return f"{dataset_prefix}_{start_year}_{month_abbr(start_month)}_{end_year}_{month_abbr(end_month)}"


def preprocess_dataset(preprocess_df: pd.DataFrame) -> pd.DataFrame:
    """Method to preprocess input data before model is trained on it

    Args:
      preprocess_df (pd.DataFrame): Input dataframe to be preprocessed

    Returns:
      pd.DataFrame: Dataframe after preprocessing
    """
    preprocess_df.columns = [x.split('__')[1].lower() if '__' in x else x.lower() for x in preprocess_df.columns]
    preprocess_df = preprocess_df.loc[:, FEATURES + EVENT_COLS + ('company_id', 'timestamp')]
    preprocess_df = preprocess_df[preprocess_df[DURATION_COL] > 0]
    preprocess_df.dropna(subset=DROP_NA_COLS, inplace=True)
    preprocess_df.reset_index(drop=True, inplace=True)
    return preprocess_df


def get_validation_months(validation_start_year: int, validation_start_month: int, validation_stop_year: int,
                          validation_stop_month: int) -> list:
    """Method to get the list of months between start and end dates

    Args:
      validation_start_year (int): Start Date - Year
      validation_start_month (int): Start Date - Month
      validation_stop_year (int): Stop Date - Year
      validation_stop_month (int): Stop Date - Year

    Returns:
      list: List of months between start and stop dates
    """
    validation_start_date = datetime.date(validation_start_year, validation_start_month, 1)
    validation_end_date = datetime.date(validation_stop_year, validation_stop_month, 1)
    list_of_months = []
    temp_date = validation_start_date
    while temp_date <= validation_end_date:
        list_of_months.append(temp_date)
        temp_date = temp_date + relativedelta(months=+1)
    return list_of_months


def get_sar_labels(feature_df=None):
    timestamp_set = list(feature_df['timestamp'].unique())
    old_date_set = [str(p).split('T')[0] for p in timestamp_set]
    new_date_set = [str(date_var + np.timedelta64(90, 'D')).split('T')[0] for date_var in timestamp_set]
    fincrime_records_set = []
    print(old_date_set, new_date_set)
    for i in range(len(old_date_set)):
        SAR_QUERY = SAR_MEMBERS_QUERY.format(old_date=old_date_set[i], new_date=new_date_set[i])
        spine_df = spark_connector(SAR_QUERY)
        df = spine_df.toPandas()
        df["timestamp"] = [old_date_set[i] for p in range(len(df))]
        fincrime_records_set.append(df)
    return fincrime_records_set


def process_sar_dataframe(fincrime_records_set=None):
    final_df = pd.DataFrame()
    for df_1 in fincrime_records_set:
        df_1['timestamp'] = df_1['timestamp'].apply(pd.Timestamp)
        df_1["duration"] = df_1['FIRST_SAR_CREATED_DT'].astype('datetime64', copy=False) - df_1['timestamp']
        df_1["duration"] = df_1["duration"].dt.days
        df_2 = df_1[["duration", "COMPANY_ID", "timestamp", 'FIRST_SAR_CREATED_DT']]
        df_2.rename(columns={'COMPANY_ID': 'company_id', 'FIRST_SAR_CREATED_DT': 'first_sar_created_dt'}, inplace=True)
        df_2.drop_duplicates(subset=['duration', 'company_id'], keep="first", inplace=True)
        final_df = final_df.append(df_2, ignore_index=True)
    return final_df


def create_target_variables(features):
    features["fincrime90_duration"] = features['first_sar_created_dt'] - features['timestamp']
    features["fincrime90_duration"] = features["fincrime90_duration"].dt.days
    features["fincrime90_duration"].fillna(90, inplace=True)
    features.loc[features["first_sar_created_dt"].isna(), 'fincrime90_event'] = 0
    features.loc[~features["first_sar_created_dt"].isna(), 'fincrime90_event'] = 1
    return features

# COMMAND ----------

