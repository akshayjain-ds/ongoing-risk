# Databricks notebook source
# MAGIC %run ./set_up

# COMMAND ----------

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
import os
import pyspark.sql.functions as F 
from pyspark.sql.functions import col
import numpy as np
from pyspark.sql.types import StringType
import gc 
from numpy import nan
from datetime import datetime, timedelta
from kneed import KneeLocator

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

# MAGIC %run ./train_metadata

# COMMAND ----------

# MAGIC %run ./queries

# COMMAND ----------

appf = spark_connector(appf_query.format(from_date=timestamp_start_date, 
                                         to_date=str(pd.to_datetime(scoring_date).date() + timedelta(days=ntt_period))))
appf = appf.toPandas()
appf.columns = [col.split("__")[1].lower() if col.__contains__("__") else col.lower() for col in appf.columns]
appf.drop_duplicates(subset = [id1, 'transaction_at', 'app_fraud_type', 'amount'], inplace=True)
appf.shape

# COMMAND ----------

appf.head()

# COMMAND ----------

fraud_company_count, fraud_company_amount = appf[id1].nunique(), appf['amount'].sum()
fraud_company_count, fraud_company_amount

# COMMAND ----------

# MAGIC %md
# MAGIC Random Day

# COMMAND ----------

random_day_df = pd.read_csv(f"/Workspace/Shared/Decisioning/Strawberries/ongoing-ntt_mp_risk-v1/artefacts/ntt_ftt_random_day_base_2022-01-01_2023-12-31.csv.gz",
                            memory_map=True, dtype={id1: str, id2: str})

random_day_df = random_day_df.assign(

  company_created_on = lambda col: pd.to_datetime(col[created_on]),
  timestamp = lambda col: pd.to_datetime(col[timestamp]),
  approved_at = lambda col:  pd.to_datetime(col['approved_at']),

  is_app_fraud_as_ntt_ftt = lambda col: col['is_app_fraud_as_ntt_ftt']
  .apply(lambda x: eval(x)),
  days_to_fraud_as_ntt_ftt = lambda col: col['days_to_fraud_as_ntt_ftt']
  .apply(lambda x: eval(x)),
  app_fraud_amount_as_ntt_ftt = lambda col: col['app_fraud_amount_as_ntt_ftt']
  .apply(lambda x: eval(x)),
  app_fraud_type_as_ntt_ftt = lambda col: col['app_fraud_type_as_ntt_ftt']
  .apply(lambda x: eval(x))
)

random_day_df.shape

# COMMAND ----------

random_day_df['is_ntt'].mean(), random_day_df['is_ftt'].mean()

# COMMAND ----------

random_day_df.head()

# COMMAND ----------

random_frauds = pd.concat([
  random_day_df[[id1, timestamp, 'is_app_fraud_as_ntt_ftt']].explode('is_app_fraud_as_ntt_ftt').reset_index(drop=True),
  random_day_df[[id1, timestamp, 'days_to_fraud_as_ntt_ftt']].explode('days_to_fraud_as_ntt_ftt').reset_index(drop=True)[['days_to_fraud_as_ntt_ftt']],
  random_day_df[[id1, timestamp, 'app_fraud_amount_as_ntt_ftt']].explode('app_fraud_amount_as_ntt_ftt').reset_index(drop=True)[['app_fraud_amount_as_ntt_ftt']],
  random_day_df[[id1, timestamp, 'app_fraud_type_as_ntt_ftt']].explode('app_fraud_type_as_ntt_ftt').reset_index(drop=True)[['app_fraud_type_as_ntt_ftt']]
], axis=1, join='inner')
random_frauds = random_frauds[random_frauds['is_app_fraud_as_ntt_ftt']]
random_frauds = random_frauds.assign(
  is_app_fraud_as_ntt_ftt = lambda col: pd.to_numeric(col['is_app_fraud_as_ntt_ftt']),
  days_to_fraud_as_ntt_ftt = lambda col: pd.to_numeric(col['days_to_fraud_as_ntt_ftt']),
  app_fraud_amount_as_ntt_ftt = lambda col: pd.to_numeric(col['app_fraud_amount_as_ntt_ftt'])
)
random_frauds.shape

# COMMAND ----------

random_frauds['days_to_fraud_as_ntt_ftt'].describe()

# COMMAND ----------

random_frauds['app_fraud_amount_as_ntt_ftt'].describe(percentiles=np.linspace(0,1,21))

# COMMAND ----------

random_frauds.sort_values(by=['days_to_fraud_as_ntt_ftt'], inplace=True)
random_frauds['cum_is_app_fraud_as_ntt_ftt'] = random_frauds['is_app_fraud_as_ntt_ftt'].cumsum()/random_frauds['is_app_fraud_as_ntt_ftt'].sum()

# COMMAND ----------

random_frauds.plot(x='days_to_fraud_as_ntt_ftt', y='cum_is_app_fraud_as_ntt_ftt', grid=True)

# COMMAND ----------

kn = KneeLocator(random_frauds['days_to_fraud_as_ntt_ftt'], 
                 random_frauds['cum_is_app_fraud_as_ntt_ftt'],
                 curve='concave', direction='increasing',
                 )
print(kn.knee)
kn.plot_knee_normalized()

# COMMAND ----------

days=45

random_frauds['days_to_fraud_as_ntt_ftt'].mean(), random_frauds[random_frauds['days_to_fraud_as_ntt_ftt']<days][id1].shape, random_frauds[random_frauds['days_to_fraud_as_ntt_ftt']<days][id1].nunique()

# COMMAND ----------

random_frauds[random_frauds['days_to_fraud_as_ntt_ftt']<days][id1].nunique()/fraud_company_count

# COMMAND ----------

random_day_df['is_app_fraud_30d'].mean(), random_day_df['is_app_fraud_45d'].mean(), random_day_df['is_app_fraud_60d'].mean(), random_day_df['is_app_fraud_90d'].mean()

# COMMAND ----------

random_day_df['days_to_fraud_30d'].mean(), random_day_df['days_to_fraud_45d'].mean(), random_day_df['days_to_fraud_60d'].mean(), random_day_df['days_to_fraud_90d'].mean()

# COMMAND ----------

random_day_df[random_day_df['days_to_fraud_30d']==0][id1].nunique(), random_day_df[random_day_df['is_app_fraud_30d']][id1].nunique(), random_day_df[random_day_df['is_app_fraud_45d']][id1].nunique(), random_day_df[random_day_df['is_app_fraud_60d']][id1].nunique(), random_day_df[random_day_df['is_app_fraud_90d']][id1].nunique()

# COMMAND ----------

random_day_df['app_fraud_amount_30d'].mean(), random_day_df['app_fraud_amount_45d'].mean(), random_day_df['app_fraud_amount_60d'].mean(), random_day_df['app_fraud_amount_90d'].mean()

# COMMAND ----------

random_day_df['app_fraud_amount_45d'].describe(np.linspace(0,1,21))

# COMMAND ----------

# MAGIC %md
# MAGIC Fixed Day

# COMMAND ----------

fixed_day_df = pd.read_csv(f"/Workspace/Shared/Decisioning/Strawberries/ongoing-ntt_mp_risk-v1/artefacts/ntt_ftt_fixed_day_base_2022-01-01_2023-12-31.csv.gz",
                            memory_map=True, dtype={id1: str, id2: str})

fixed_day_df = fixed_day_df.assign(

  company_created_on = lambda col: pd.to_datetime(col[created_on]),
  timestamp = lambda col: pd.to_datetime(col[timestamp]),
  approved_at = lambda col:  pd.to_datetime(col['approved_at']),

  is_app_fraud_as_ntt_ftt = lambda col: col['is_app_fraud_as_ntt_ftt']
  .apply(lambda x: eval(x)),
  days_to_fraud_as_ntt_ftt = lambda col: col['days_to_fraud_as_ntt_ftt']
  .apply(lambda x: eval(x)),
  app_fraud_amount_as_ntt_ftt = lambda col: col['app_fraud_amount_as_ntt_ftt']
  .apply(lambda x: eval(x)),
  app_fraud_type_as_ntt_ftt = lambda col: col['app_fraud_type_as_ntt_ftt']
  .apply(lambda x: eval(x))    
)

fixed_day_df.shape

# COMMAND ----------

fixed_day_df['is_ntt'].mean(), fixed_day_df['is_ftt'].mean()

# COMMAND ----------

fixed_day_df.head()

# COMMAND ----------

fixed_frauds = pd.concat([
  fixed_day_df[[id1, timestamp, 'is_app_fraud_as_ntt_ftt']].explode('is_app_fraud_as_ntt_ftt').reset_index(drop=True),
  fixed_day_df[[id1, timestamp, 'days_to_fraud_as_ntt_ftt']].explode('days_to_fraud_as_ntt_ftt').reset_index(drop=True)[['days_to_fraud_as_ntt_ftt']],
  fixed_day_df[[id1, timestamp, 'app_fraud_amount_as_ntt_ftt']].explode('app_fraud_amount_as_ntt_ftt').reset_index(drop=True)[['app_fraud_amount_as_ntt_ftt']],
  fixed_day_df[[id1, timestamp, 'app_fraud_type_as_ntt_ftt']].explode('app_fraud_type_as_ntt_ftt').reset_index(drop=True)[['app_fraud_type_as_ntt_ftt']]
], axis=1, join='inner')
fixed_frauds = fixed_frauds[fixed_frauds['is_app_fraud_as_ntt_ftt']]
fixed_frauds = fixed_frauds.assign(
  
  is_app_fraud_as_ntt_ftt = lambda col: pd.to_numeric(col['is_app_fraud_as_ntt_ftt']),
  days_to_fraud_as_ntt_ftt = lambda col: pd.to_numeric(col['days_to_fraud_as_ntt_ftt']),
  app_fraud_amount_as_ntt_ftt = lambda col: pd.to_numeric(col['app_fraud_amount_as_ntt_ftt'])
)
fixed_frauds.shape

# COMMAND ----------

fixed_frauds['days_to_fraud_as_ntt_ftt'].describe()

# COMMAND ----------

np.linspace(0,1,21)

# COMMAND ----------

fixed_frauds['app_fraud_amount_as_ntt_ftt'].describe(percentiles=np.linspace(0,1,21))

# COMMAND ----------

fixed_frauds.sort_values(by=['days_to_fraud_as_ntt_ftt'], inplace=True)
fixed_frauds['cum_is_app_fraud_as_ntt_ftt'] = fixed_frauds['is_app_fraud_as_ntt_ftt'].cumsum()/fixed_frauds['is_app_fraud_as_ntt_ftt'].sum()

# COMMAND ----------

fixed_frauds.plot(x='days_to_fraud_as_ntt_ftt', y='cum_is_app_fraud_as_ntt_ftt', grid=True)

# COMMAND ----------

kn = KneeLocator(fixed_frauds['days_to_fraud_as_ntt_ftt'], 
                 fixed_frauds['cum_is_app_fraud_as_ntt_ftt'],
                 curve='concave', direction='increasing'
                 )
print(kn.knee)
kn.plot_knee_normalized()

# COMMAND ----------

days=45

fixed_frauds['days_to_fraud_as_ntt_ftt'].mean(), fixed_frauds[fixed_frauds['days_to_fraud_as_ntt_ftt']<days][id1].shape, fixed_frauds[fixed_frauds['days_to_fraud_as_ntt_ftt']<days][id1].nunique()

# COMMAND ----------

fixed_frauds[fixed_frauds['days_to_fraud_as_ntt_ftt']<days][id1].nunique()/fraud_company_count

# COMMAND ----------

fixed_day_df['is_app_fraud_30d'].mean(), fixed_day_df['is_app_fraud_45d'].mean(), fixed_day_df['is_app_fraud_60d'].mean(), fixed_day_df['is_app_fraud_90d'].mean()

# COMMAND ----------

fixed_day_df['days_to_fraud_30d'].mean(), fixed_day_df['days_to_fraud_45d'].mean(), fixed_day_df['days_to_fraud_60d'].mean(), fixed_day_df['days_to_fraud_90d'].mean()

# COMMAND ----------

fixed_day_df[fixed_day_df['days_to_fraud_30d']==0][id1].nunique(), fixed_day_df[fixed_day_df['is_app_fraud_30d']][id1].nunique(), fixed_day_df[fixed_day_df['is_app_fraud_45d']][id1].nunique(), fixed_day_df[fixed_day_df['is_app_fraud_60d']][id1].nunique(), fixed_day_df[fixed_day_df['is_app_fraud_90d']][id1].nunique()

# COMMAND ----------

fixed_day_df['app_fraud_amount_30d'].mean(), fixed_day_df['app_fraud_amount_45d'].mean(), fixed_day_df['app_fraud_amount_60d'].mean(), fixed_day_df['app_fraud_amount_90d'].mean()

# COMMAND ----------

fixed_day_df['app_fraud_amount_45d'].describe(percentiles =np.linspace(0,1,21))

# COMMAND ----------

gc.collect()

# COMMAND ----------

feature_list = ['days_on_books',
'is_ntt',
'is_ftt',
'days_to_transact',
'days_remaining_as_ntt_ftt',
'avg_deposit',
'avg_withdrawal',
'card_pans_cnt',
'cardpmt_acceptors',
'cardpmt_wtd_pct',
'cardwtd_wtd_pct',
'cash_dep_pct',
'cashtxns_latehrs',
'ddebit_beneficiaries',
'ddebit_wtd_pct',
'deposit_wtd_frequency_ratio',
'fastpmt_benefactors',
'fastpmt_beneficiaries',
'fastpmt_dep_pct',
'fastpmt_wtd_pct',
'high_card_pmts',
'high_card_wtds',
'high_ddebit',
'high_fpmt_in',
'high_fpmt_out',
'high_pmt_in',
'high_pmt_out',
'hmrc_txns_cnt',
'inpmt_dep_pct',
'max_cash_deposits',
'max_deposit',
'max_withdrawal',
'outpmt_wtd_pct',
'pct_round_txns',
'pct_unique_txns',
'pos_atm_locations',
'tester_pmt_cnt',
'xero_txns_cnt',
'company_age_at_timestamp',
'receipt_uploaded_before_timestamp',
'receipt_match_before_timestamp',
'first_invoice_before_timestamp',
'invoice_matched_before_timestamp',
'invoice_chased_before_timestamp',
'activity_before_timestamp',
'login_before_timestamp',
'applicant_email_numeric',
'applicant_age_at_completion',
'days_to_approval',
'applicant_years_to_id_expiry',
'is_restricted_keyword_present',
'company_is_registered']
len(feature_list)

# COMMAND ----------

fixed_day_features = pd.read_csv("/Workspace/Shared/Decisioning/Strawberries/ongoing-ntt_mp_risk-v1/artefacts/ntt_ftt_fixed_day_raw_features_2022-01-01_2023-12-31.csv.gz", 
                                 memory_map=True, dtype={id1: str, id2: str})
fixed_day_features = fixed_day_features.assign(
  timestamp = lambda col: pd.to_datetime(col[timestamp])
)
fixed_day_features = fixed_day_df[[id1, timestamp] + ['is_app_fraud_45d']].merge(fixed_day_features[[id1, timestamp] + feature_list], on=[id1, timestamp], how='inner')
fixed_day_features.shape

# COMMAND ----------

summary = fixed_day_features.groupby(['is_app_fraud_45d'])[feature_list].mean()
summary.T

# COMMAND ----------

random_day_features = pd.read_csv("/Workspace/Shared/Decisioning/Strawberries/ongoing-ntt_mp_risk-v1/artefacts/ntt_ftt_random_day_raw_features_2022-01-01_2023-12-31.csv.gz", 
                                 memory_map=True, dtype={id1: str, id2: str})
random_day_features = random_day_features.assign(
  timestamp = lambda col: pd.to_datetime(col[timestamp])
)
random_day_features = random_day_df[[id1, timestamp] + ['is_app_fraud_45d']].merge(random_day_features[[id1, timestamp] + feature_list], on=[id1, timestamp], how='inner')
random_day_features.shape

# COMMAND ----------

summary = random_day_features.groupby(['is_app_fraud_45d'])[feature_list].mean()
summary.T

# COMMAND ----------

appf = spark_connector(appf_query.format(from_date=timestamp_start_date, 
                                         to_date=str(pd.to_datetime(scoring_date).date() + timedelta(days=days))))
appf = appf.toPandas()
appf.columns = [col.split("__")[1].lower() if col.__contains__("__") else col.lower() for col in appf.columns]
appf.drop_duplicates(subset = [id1, 'transaction_at', 'app_fraud_type', 'amount'], inplace=True)
appf.shape

# COMMAND ----------

absent_companies = set(appf[id1].tolist()) - set(random_day_df[random_day_df['is_app_fraud_45d']==1][id1].unique().tolist())
appf[appf[id1].isin(absent_companies)][id1].nunique()

# COMMAND ----------

absent_companies_df = appf[appf[id1].isin(absent_companies)]
absent_companies_df

# COMMAND ----------

absent_companies = set(appf[id1].tolist()) - set(fixed_day_df[fixed_day_df['is_app_fraud_45d']==1][id1].unique().tolist())
appf[appf[id1].isin(absent_companies)][id1].nunique()

# COMMAND ----------

absent_companies_df = appf[appf[id1].isin(absent_companies)]
absent_companies_df

# COMMAND ----------


