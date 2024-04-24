# Databricks notebook source
# MAGIC %md #### Install dependencies

# COMMAND ----------

# %pip install catboost==0.26.1
# %pip install mlflow==1.25.0
# # %pip install shap==0.36.0
# %pip install numpy==1.22.4
# %pip install matplotlib==3.2
# # %pip install tecton==0.6.10
# %pip install category_encoders==2.6.1
# %pip install pandas==1.4.4
# # %pip install dill==0.3.6
# # %pip install pandas-profiling==3.3.0
# %pip install cloudpickle==2.2.1

# GANs and LLMs related packages
%pip install torch==1.13.1
%pip install peft
%pip install accelerate -U
%pip install -U transformers
%pip install datasets
%pip install evaluate
%pip install lime
%pip install mlflow
# %pip install tabgan

# COMMAND ----------

# MAGIC %md #### Importing Library

# COMMAND ----------

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql.types import DoubleType
# from tecton import DatabricksClusterConfig
# from tecton import Aggregation
# from tecton import const
# from tecton import FilteredSource
# from tecton import stream_feature_view
# from tecton import transformation
# from tecton import Entity

# from tecton import batch_feature_view

# import tecton

# COMMAND ----------

import functools
import time
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import when, col
from pyspark.sql.types import *
from datetime import date, datetime
import pandas as pd
import numpy as np
import itertools
import logging
# import tecton
import os
import gc
import tempfile
from pyspark.sql.functions import struct
import matplotlib.pyplot as plt
# import shap
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
from dateutil.relativedelta import *
# from pandas_profiling import ProfileReport
import re
from sklearn.pipeline import (make_pipeline, Pipeline)
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import (StandardScaler, OneHotEncoder)
from sklearn.base import BaseEstimator, TransformerMixin
# from category_encoders.target_encoder import TargetEncoder
# from category_encoders.cat_boost import CatBoostEncoder
import json
import seaborn as sns
import dateutil.parser
# from numba import njit
# %matplotlib inline
import warnings

warnings.filterwarnings("ignore")

SCOPE = "tecton"
SNOWFLAKE_DATABASE = "TIDE"
SNOWFLAKE_SOURCE_DATABASE = 'TIDE'
# SNOWFLAKE_USER = dbutils.secrets.get(scope=SCOPE, key="SNOWFLAKE_DATABRICKS_USER")
SNOWFLAKE_USER = "DATABRICKS"
SNOWFLAKE_PASSWORD = dbutils.secrets.get(scope=SCOPE, key="SNOWFLAKE_DATABRICKS_PASSWORD")
SNOWFLAKE_ACCOUNT = dbutils.secrets.get(scope=SCOPE, key="SNOWFLAKE_DATABRICKS_ACCOUNT")
# SNOWFLAKE_WAREHOUSE = "TECTON_WH"
SNOWFLAKE_SCHEMA = "DB_TIDE"
SNOWFLAKE_WAREHOUSE = "DATABRICKS_WH"
SNOWFLAKE_ROLE = "DATABRICKS_ROLE"

# snowflake connection options
CONNECTION_OPTIONS = dict(sfUrl=SNOWFLAKE_ACCOUNT,
                          sfUser=SNOWFLAKE_USER,
                          sfPassword=SNOWFLAKE_PASSWORD,
                          sfDatabase=SNOWFLAKE_DATABASE,
                          sfSchema=SNOWFLAKE_SCHEMA,
                          sfWarehouse=SNOWFLAKE_WAREHOUSE,
                          sfRole=SNOWFLAKE_ROLE, )


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


def save_dataset(output_df: DataFrame, dataset_name: str, overwrite=False):
    if overwrite:
        output_df.write.format("parquet").mode("overwrite").save(dataset_name)
    else:
        output_df.write.format("parquet").save(dataset_name)


def load_dataset(dataset_name: str):
    return spark.read.parquet(dataset_name)


def file_exists(dataset_name: str):
    try:
        dbutils.fs.ls(dataset_name)
        return True
    except Exception as e:
        if 'java.io.FileNotFoundException' in str(e):
            return False
        else:
            raise


def time_diff_in_seconds(timestamps):
    possible_formats = ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S']
    diffs = []
    for i in range(len(timestamps) - 1):
        t1, t2 = None, None
        for fmt in possible_formats:
            try:
                t1 = datetime.strptime(timestamps[i], fmt)
                t2 = datetime.strptime(timestamps[i + 1], fmt)
                break
            except ValueError:
                pass
        if t1 is not None and t2 is not None:
            diff = float(round(t2.timestamp() - t1.timestamp(), 3))
            diffs.append(diff)
    return np.mean(diffs)


# COMMAND ----------

INPUT_COLS = (
    '2hr_cash_inflow', '4hr_cash_inflow', '8hr_cash_inflow', '12hr_cash_inflow', '24hr_cash_inflow', '48hr_cash_inflow',
    '2hr_cash_outflow', '4hr_cash_outflow', '8hr_cash_outflow', '12hr_cash_outflow', '24hr_cash_outflow',
    '48hr_cash_outflow',
    '60days_average_deposit_value', '60days_inbound_payments_count', '60days_outgoing_payments_over_threshold_count',
    'requested_payment_value', 'is_registered', 'risk_score', 'request_timestamp', 'is_clearlisted',
    'recipient_account_number',
    'recipient_sort_code', 'last_whitelisted_at', 'credit_line_amount', 'credit_line_received_at', 'risk_band',
    'event_ignore_expiration', 'tm_ignore_expiration_at'
)
FEATURE_COLS = (
    '2hr_cash_inflow', '4hr_cash_inflow', '8hr_cash_inflow', '12hr_cash_inflow', '24hr_cash_inflow',
    '48hr_cash_inflow', '2hr_cash_outflow', '4hr_cash_outflow', '8hr_cash_outflow', '12hr_cash_outflow',
    '24hr_cash_outflow', '48hr_cash_outflow', '60days_average_deposit_value', '60days_inbound_payments_count',
    '60days_outgoing_payments_over_threshold_count', 'requested_payment_value', 'is_registered', 'risk_score'
)
STR_COLS = ('recipient_account_number', 'recipient_sort_code', 'request_timestamp', 'risk_band')
BOOLEAN_COLS = ('is_clearlisted', 'event_ignore_expiration')
NUMERIC_COLS = tuple([x for x in INPUT_COLS if x not in STR_COLS and x not in BOOLEAN_COLS])
NOT_NULL_COLS = ('requested_payment_value', 'is_registered')
TARGET_COL = 'is_app_victim_counter_party'
COMPANY_COL = 'company_id'
USE_SEPERATE_ENCODING = True
PAYMENT_MATCHING_COLS = ('company_id', 'requested_payment_value', 'request_timestamp')

# COMMAND ----------

EVENT_SWITCH_DATE = '2021-07-01'

GET_CLEARED_TXNS_QUERY_DC = '''
WITH txn_events AS (
    SELECT "timestamp", CAST(TRY_PARSE_JSON(PAYLOAD):accountId AS VARCHAR) as accountId, 
      COALESCE(RIGHT(TRIM(TRY_PARSE_JSON(PAYLOAD):creditorAccount:identification), 8), '-') AS destination_account_id, 
      CAST(TRY_PARSE_JSON(PAYLOAD):transactionId AS VARCHAR) as transactionId,
      COALESCE(LEFT(TRIM(TRY_PARSE_JSON(PAYLOAD):creditorAccount:identification), 6), '-') AS destination_account_sort_code, 
      CASE WHEN event_type = 'application/vnd.tide.transaction-received.v1' THEN CAST(TRY_PARSE_JSON(PAYLOAD):amount:value AS FLOAT)/100.0 
           ELSE CAST(TRY_PARSE_JSON(PAYLOAD):amount:amount AS FLOAT)/100.0 END 
           as requested_payment_value
    FROM TIDE.EVENT_SERVICE.EVENT
    WHERE "timestamp" > '{START_DATE}' AND "timestamp" < '{END_DATE}' AND 
          ((event_type = 'application/vnd.tide.transaction-received.v1' AND "timestamp" < '{SWITCH_DATE}') OR 
          (event_type = 'application/vnd.tide.transaction-created.v1.0' AND "timestamp" >= '{SWITCH_DATE}')) AND 
          CHECK_JSON(PAYLOAD) IS NULL AND TRY_PARSE_JSON(PAYLOAD):proprietaryBankTransactionCode.code IN ('DOMESTIC_TRANSFER', 'INTERNAL_BOOK_TRANSFER') AND 
          TRY_PARSE_JSON(PAYLOAD):status = 'CLEARED'
  )
  SELECT T."timestamp" as timestamp, CAST(A.COMPANYID AS VARCHAR) as company_id, T.ACCOUNTID as account_id, T.ACCOUNTID as source_account_id,
          T.destination_account_id, T.destination_account_sort_code, M.RISK_RATING as risk_score, T.requested_payment_value, T.TRANSACTIONID, TXN.transactiontype, TXN.txnref
  FROM txn_events T JOIN TIDE.DB_TIDE.ACCOUNT A ON T.accountId = CAST(A.ACCOUNTID AS VARCHAR) 
                    JOIN TIDE.DB_TIDE.COMPANY C ON CAST(A.COMPANYID AS VARCHAR) = CAST(C.COMPANYID AS VARCHAR) 
                    JOIN TIDE.DB_TIDE.MEMBERSHIP M ON M.ID = C.MEMBERSHIP_ID
                    JOIN TIDE.DB_TIDE.TRANSACTION TXN ON T.TRANSACTIONID = TXN.TRANSACTIONID
'''

GET_CLEARED_TXNS_QUERY_DEBIT = '''
WITH txn_events AS (
    SELECT "timestamp", CAST(TRY_PARSE_JSON(PAYLOAD):accountId AS VARCHAR) as accountId, 
      COALESCE(RIGHT(TRIM(TRY_PARSE_JSON(PAYLOAD):creditorAccount:identification), 8), '-') AS destination_account_id, 
      CAST(TRY_PARSE_JSON(PAYLOAD):transactionId AS VARCHAR) as transactionId,
      COALESCE(LEFT(TRIM(TRY_PARSE_JSON(PAYLOAD):creditorAccount:identification), 6), '-') AS destination_account_sort_code, 
      CAST(TRY_PARSE_JSON(PAYLOAD):creditorAccount:identification as VARCHAR) as destination_account_identification,
      CAST(TRY_PARSE_JSON(PAYLOAD):debtorAccount:identification as VARCHAR) as source_account_identification,
      TRY_PARSE_JSON(PAYLOAD):proprietaryBankTransactionCode:issuer as issuer,
      CASE WHEN event_type = 'application/vnd.tide.transaction-received.v1' THEN CAST(TRY_PARSE_JSON(PAYLOAD):amount:value AS FLOAT)/100.0 
           ELSE CAST(TRY_PARSE_JSON(PAYLOAD):amount:amount AS FLOAT)/100.0 END 
           as requested_payment_value
    FROM TIDE.EVENT_SERVICE.EVENT
    WHERE "timestamp" > '{START_DATE}' AND "timestamp" < '{END_DATE}' AND 
          ((event_type = 'application/vnd.tide.transaction-received.v1' AND "timestamp" < '{SWITCH_DATE}') OR 
          (event_type = 'application/vnd.tide.transaction-created.v1.0' AND "timestamp" >= '{SWITCH_DATE}')) AND 
          CHECK_JSON(PAYLOAD) IS NULL AND TRY_PARSE_JSON(PAYLOAD):proprietaryBankTransactionCode.code IN ('DOMESTIC_TRANSFER', 'INTERNAL_BOOK_TRANSFER') AND 
          TRY_PARSE_JSON(PAYLOAD):proprietaryBankTransactionCode:issuer = 'TIDE' AND
          TRY_PARSE_JSON(PAYLOAD):status = 'CLEARED' AND TRY_PARSE_JSON(PAYLOAD):creditDebitIndicator = 'DEBIT'
  )
  SELECT T."timestamp" as timestamp, CAST(A.COMPANYID AS VARCHAR) as company_id, T.ACCOUNTID as account_id, T.ACCOUNTID as source_account_id,
          T.destination_account_id, T.destination_account_sort_code, M.RISK_RATING as risk_score, T.requested_payment_value, T.TRANSACTIONID, TXN.transactiontype, TXN.txnref, T.destination_account_identification, T.source_account_identification, T.issuer
  FROM txn_events T JOIN TIDE.DB_TIDE.ACCOUNT A ON T.accountId = CAST(A.ACCOUNTID AS VARCHAR) 
                    JOIN TIDE.DB_TIDE.COMPANY C ON CAST(A.COMPANYID AS VARCHAR) = CAST(C.COMPANYID AS VARCHAR) 
                    JOIN TIDE.DB_TIDE.MEMBERSHIP M ON M.ID = C.MEMBERSHIP_ID
                    JOIN TIDE.DB_TIDE.TRANSACTION TXN ON T.TRANSACTIONID = TXN.TRANSACTIONID
'''

GET_CLEARED_TXNS_QUERY_CREDIT = '''
WITH txn_events AS (
    SELECT "timestamp", CAST(TRY_PARSE_JSON(PAYLOAD):accountId AS VARCHAR) as accountId, 
      COALESCE(RIGHT(TRIM(TRY_PARSE_JSON(PAYLOAD):creditorAccount:identification), 8), '-') AS destination_account_id, 
      CAST(TRY_PARSE_JSON(PAYLOAD):transactionId AS VARCHAR) as transactionId,
      COALESCE(LEFT(TRIM(TRY_PARSE_JSON(PAYLOAD):creditorAccount:identification), 6), '-') AS destination_account_sort_code, 
      CASE WHEN event_type = 'application/vnd.tide.transaction-received.v1' THEN CAST(TRY_PARSE_JSON(PAYLOAD):amount:value AS FLOAT)/100.0 
           ELSE CAST(TRY_PARSE_JSON(PAYLOAD):amount:amount AS FLOAT)/100.0 END 
           as requested_payment_value
    FROM TIDE.EVENT_SERVICE.EVENT
    WHERE "timestamp" > '{START_DATE}' AND "timestamp" < '{END_DATE}' AND 
          ((event_type = 'application/vnd.tide.transaction-received.v1' AND "timestamp" < '{SWITCH_DATE}') OR 
          (event_type = 'application/vnd.tide.transaction-created.v1.0' AND "timestamp" >= '{SWITCH_DATE}')) AND 
          CHECK_JSON(PAYLOAD) IS NULL AND TRY_PARSE_JSON(PAYLOAD):proprietaryBankTransactionCode.code IN ('DOMESTIC_TRANSFER', 'INTERNAL_BOOK_TRANSFER') AND 
          TRY_PARSE_JSON(PAYLOAD):status = 'CLEARED' AND TRY_PARSE_JSON(PAYLOAD):creditDebitIndicator = 'CREDIT'
  )
  SELECT T."timestamp" as timestamp, CAST(A.COMPANYID AS VARCHAR) as company_id, T.ACCOUNTID as account_id, T.ACCOUNTID as source_account_id,
          T.destination_account_id, T.destination_account_sort_code, M.RISK_RATING as risk_score, T.requested_payment_value, T.TRANSACTIONID, TXN.transactiontype, TXN.txnref
  FROM txn_events T JOIN TIDE.DB_TIDE.ACCOUNT A ON T.accountId = CAST(A.ACCOUNTID AS VARCHAR) 
                    JOIN TIDE.DB_TIDE.COMPANY C ON CAST(A.COMPANYID AS VARCHAR) = CAST(C.COMPANYID AS VARCHAR) 
                    JOIN TIDE.DB_TIDE.MEMBERSHIP M ON M.ID = C.MEMBERSHIP_ID
                    JOIN TIDE.DB_TIDE.TRANSACTION TXN ON T.TRANSACTIONID = TXN.TRANSACTIONID
'''

SAR_COMPANIES_QUERY = """
  SELECT  REGEXP_SUBSTR(TRIM(jira_tickets.ticket_summary),'[0-9a-fA-F]{{8}}\-[0-9a-fA-F]{{4}}\-[0-9a-fA-F]{{4}}\-[0-9a-fA-F]{{4}}\-[0-9a-fA-F]{{12}}|[0-9]{{4,}}') AS company_id, TRIM(jira_tickets.issue_type) AS issue_type,
          TO_DATE(DATE_TRUNC('DAY', jira_tickets.TICKET_CREATED_AT)) AS ticket_created_date, TO_DATE(DATE_TRUNC('DAY', MIN(jira_ticket_changes.change_at))) AS sar_created_date
  FROM    TIDE.PRES_JIRA.JIRA_TICKETS AS jira_tickets LEFT JOIN TIDE.PRES_JIRA.JIRA_TICKET_CHANGES AS jira_ticket_changes ON jira_tickets.TICKET_ID = jira_ticket_changes.TICKET_ID
  WHERE   jira_tickets.PROJECT_KEY = 'RCM' AND TRIM(jira_tickets.issue_type) IN ('TM alert', 'Risk case') AND 
          (jira_tickets.JIRA_TICKET_STATUS IS NULL OR jira_tickets.JIRA_TICKET_STATUS <> 'Duplicates') AND 
          (NOT (jira_tickets.is_subtask = 1 ) OR (jira_tickets.is_subtask = 1 ) IS NULL) AND 
          jira_ticket_changes.NEW_VALUE IN ('SAR', 'Tide Review', 'PPS Review', 'Submit to NCA', 'NCA Approval', 'NCA Refusal', 'Clear funds', 'Off-board customer')
  GROUP BY 1, 2, 3 HAVING ticket_created_date >= '{ALERT_START_DATE}' AND ticket_created_date < '{ALERT_END_DATE}' AND sar_created_date < '{SAR_LAST_DATE}'
"""

VICTIM_QUERY = """
with app as 
(select ticket_key, REGEXP_REPLACE(txn_reference, '[\\s,"\]', '') as txn_reference
from (
select ticket_key, transaction_ref_one as txn_reference
from pres_jira.jira_tickets
union 
select ticket_key, transaction_ref_two 
from pres_jira.jira_tickets
union 
select ticket_key, transaction_ref_three
from pres_jira.jira_tickets
union 
select ticket_key, transaction_ref_four 
from pres_jira.jira_tickets
union 
select ticket_key, transaction_ref_five 
from pres_jira.jira_tickets
)
where txn_reference is not null
)


, jira2 as
(
select 
distinct
coalesce(a.company_id,b.company_id) as company_id,
case when a.parent_key is not null then a.parent_key else a.ticket_key end as ticket_key,
coalesce(a.app_date_of_report,b.app_date_of_report) date_of_report,
coalesce( a.app_type_of_app_fraud, b.app_type_of_app_fraud ) AS type_of_app_fraud, 
c.txn_ref as txnref
, abs(c.amount) as amount
, c.transaction_at
, d.sortcode
, d.bank_name
from pres_jira.jira_tickets a 
left join app on a.ticket_key = app.ticket_key
left join pres_jira.jira_tickets b on a.parent_key = b.ticket_key
left join pres_core.companies e on a.company_id = e.company_id
left join pres_core.cleared_transactions c on trim(txn_reference) = c.txn_ref
left join pres_core.payment d on c.txn_ref = d.transactionreference
where a.project_key = 'AC'
qualify row_number() over(partition by txn_ref order by a.ticket_created_at asc) = 1 
)

select *,
case when type_of_app_fraud IN ('Invoice Interception Scam') THEN 'Invoice interception scam'
    when type_of_app_fraud IN ('Impersonation Scam') THEN 'Impersonation scam'
    when type_of_app_fraud IN ('Safe Account Scam') THEN 'Safe account scam'
    when type_of_app_fraud IN ('Investment Scam') THEN 'Investment scam'
    when type_of_app_fraud IN ('Purchase Scam') THEN 'Purchase scam'
    when type_of_app_fraud IN ('Other') THEN 'Unknown scam'
    else type_of_app_fraud end as mapped_fraud_type
from jira2 
where date_of_report between '{ALERT_START_DATE}' and '{ALERT_END_DATE}'
    and mapped_fraud_type in ('Invoice interception scam', 'Impersonation scam', 'Safe account scam', 'Investment scam', 'Purchase scam', 'Unknown scam')
"""

PERPETRATOR_QUERY = """
WITH DATA as (
SELECT 
    jira_fincrime.ticket_key,
    jira_fincrime.company_id,
    jira_fincrime.ticket_id,
    min(CASE WHEN jira_ticket_changes.change_at IS NOT NULL THEN jira_ticket_changes.change_at ELSE jira_fincrime.ticket_created_at END) as reported_date
FROM pres_jira.jira_tickets  AS jira_fincrime
LEFT JOIN pres_jira.jira_ticket_changes jira_ticket_changes ON jira_fincrime.ticket_id = jira_ticket_changes.ticket_id 
    AND jira_ticket_changes.change_field = 'Number of fraud reports'
WHERE jira_fincrime.project_key = 'RCM' 
    AND jira_fincrime.number_of_fraud_report IS NOT NULL
    AND jira_fincrime.is_subtask <> 1 
    AND fraud_type_new  IN ('Invoice scams', 'Mail boxes and multiple post redirections', 'Safe account scam', 'Mule', 'Telephone banking scam', 'HMRC scam', 'Impersonation scam', 'Investment scam', 'Cryptocurrency investment fraud', 'Advance fee fraud', '419 emails and letters', 'Romance scam', 'Purchase scam', 'Other', 'Not provided')
    AND (DATE(CASE WHEN jira_ticket_changes.change_at IS NOT NULL THEN jira_ticket_changes.change_at ELSE jira_fincrime.ticket_created_at END) 
        BETWEEN DATE('{ALERT_START_DATE}') AND DATE('{ALERT_END_DATE}') )
    AND jira_fincrime.jira_ticket_status not in ('Duplicate', 'Duplicates')
group by 1,2,3
HAVING reported_date between '{ALERT_START_DATE}' and '{ALERT_END_DATE}'
)
, txns as (
select ticket_key,REGEXP_REPLACE(txn_ref_all, '[\\s,"\]', '') as txn_ref_all
from (

SELECT ticket_key, trim(one_fraud_report_transaction_reference) AS txn_ref_all FROM pres_jira.jira_tickets
UNION 
SELECT ticket_key,  trim(two_fraud_report_transaction_reference) FROM pres_jira.jira_tickets
UNION 
SELECT ticket_key, trim(three_fraud_report_transaction_reference) FROM pres_jira.jira_tickets
UNION 
SELECT ticket_key,  trim(four_fraud_report_transaction_reference) FROM pres_jira.jira_tickets
UNION 
SELECT ticket_key,  trim(five_fraud_report_transaction_reference) FROM pres_jira.jira_tickets
UNION 
SELECT ticket_key, trim(six_fraud_report_transaction_reference) FROM pres_jira.jira_tickets
UNION 
SELECT ticket_key, trim(seven_fraud_report_transaction_reference) FROM pres_jira.jira_tickets
UNION 
SELECT ticket_key, trim(eight_fraud_report_transaction_reference) FROM pres_jira.jira_tickets
UNION 
SELECT ticket_key, trim(nine_fraud_report_transaction_reference) FROM pres_jira.jira_tickets
UNION 
SELECT ticket_key, trim(ten_fraud_report_transaction_reference) FROM pres_jira.jira_tickets

)
where txn_ref_all is not null

)

, tickets as
(
select distinct
    coalesce(a.company_id,b.company_id) as company_id,
    case when a.parent_key is  null then  a.ticket_key else a.parent_key end as ticket_key,
    case when a.parent_key is  null then a.fraud_type_new else sub.fraud_type_new end as fraud_type_new, 
    b.txn_ref as txnref,
    b.transaction_at,
    b.amount,
    comp.is_registered,
    comp.first_transaction_at,
    comp.industry_classification_description,
    mem.approved_at_clean,
    mem.last_touch_attribution_marketing_channel,
    mem.risk_rating,
    mem.risk_band,
    comp.is_cofo,
    CASE WHEN mem.manual_approval_triggers <> ''  THEN 'Yes' ELSE 'No' END as manual_kyc,
    incorporation_at
from 
(select * from pres_jira.jira_tickets where project_key = 'RCM') a 
left join txns t on a.ticket_key = t.ticket_key
left join pres_jira.jira_tickets sub on a.parent_key = sub.ticket_key
join pres_core.cleared_transactions b on t.txn_ref_all = b.txn_ref
left join pres_core.companies comp on b.company_id = comp.company_id
left join pres_core.memberships mem on comp.member_id = mem.member_id
where year(transaction_at) = 2022
qualify row_number() over(partition by txn_ref order by a.ticket_created_at asc) = 1 )

, jira as
(select  t.* 
from tickets t
join  
data d 
on  t.ticket_key = d.ticket_key 
)
select *, --count( distinct ticket_key ) as no_of_tickets
case when fraud_type_new IN ('Invoice scams', 'Mail boxes and multiple post redirections') THEN 'Invoice and Mandate scam'
     when fraud_type_new IN ('Safe account scam', 'Mule') THEN 'Safe account scam'  
     when fraud_type_new IN ('HMRC scam', 'Telephone banking scam', 'Impersonation scam') THEN 'Impersonation scam'
     when fraud_type_new IN ('Investment scam', 'Cryptocurrency investment fraud') THEN 'Investment scam'
     when fraud_type_new IN ('Advance fee fraud', '419 emails and letters') THEN 'Advance fee scam'
     when fraud_type_new IN ('Romance scam') THEN 'Romance scam'
     when fraud_type_new IN ('Purchase scam') THEN 'Purchase scam'
     when fraud_type_new IN ('Other', 'Not provided') THEN 'Unknown scam'
     else fraud_type_new end as mapped_fraud_type
from jira
where mapped_fraud_type in ('Invoice and Mandate scam', 'Impersonation scam', 'Safe account scam', 'Investment scam', 'Advance fee scam', 'Romance scam', 'Purchase scam', 'Unknown scam');
"""


# COMMAND ----------

def setup_dataset(dbfs_prefix: str, feature_service: str, txn_start_date: str, txn_end_date: str, sar_end_date: str,
                  recreate_file: str) -> str:
    """Method to call setup_dataset notebook

    Args:
      txn_start_date (str): Start Date from which features are calculated
      txn_end_date (str): End Date until when features are calculated
      sar_end_date (str): Cutoff Date for updates on alerts

    Returns:
      str: Name of the dataset created
    """
    input_args = {
        'dbfs_prefix': dbfs_prefix,
        'feature_service': feature_service,
        'txn_start_date': txn_start_date,
        'txn_end_date': txn_end_date,
        'sar_end_date': sar_end_date,
        'recreate_file': recreate_file
    }
    return dbutils.notebook.run(path="setup_dataset", timeout_seconds=30000, arguments=input_args)


# COMMAND ----------

FEATURE_MAPPER = {
    'company_id': 'company_id',
    'requested_payment_value': 'requested_payment_value',
    'timestamp': 'rule_feature_request_timestamp',
    'destination_account_id': 'destination_account_number',
    'destination_account_sort_code': 'destination_account_sort_code',
    'is_registered': 'is_registered_company',
    'company_type': 'registered_company_type',
    'kyc_risk_band': 'rule_feature_kyc_risk_band',
    'source_dest_clearlisting_features.is_clearlisted': 'rule_feature_is_source_dest_clearlisted',
    'source_dest_clearlisting_features.expiry_ts': 'rule_feature_source_dest_expiry_ts',
    'source_dest_clearlisting_features.amount': 'rule_feature_source_dest_amount',
    'whitelisting_features.is_clearlisted': 'rule_feature_is_clearlisted',
    'whitelisting_features.event_ignore_expiration': 'rule_feature_event_ignore_expiration',
    'whitelisting_features.last_updated_at': 'rule_feature_clearlisting_last_updated_at',
    'permanent_whitelisting_features.last_whitelisted_at': 'rule_feature_permenant_clearlisting_last_whitelisted_at',
    'credit_line_features.amount': 'rule_feature_credit_line_amount',
    'credit_line_features.last_updated_at': 'rule_feature_credit_line_last_updated_at',
    'ifre_member_features.age_at_completion': 'member_age_at_onboarding',
    'ifre_member_features.applicant_postcode': 'member_postcode',
    'ifre_member_features.applicant_id_country_issue': 'member_id_country_issue',
    'ifre_member_features.applicant_id_type': 'member_id_type',
    'ifre_member_features.industry_classification': 'registered_company_industry_classification',
    'company_transaction_features.avg_deposit': 'average_deposit_value_1y',
    'company_transaction_features.avg_withdrawal': 'average_withdrawal_value_1y',
    'company_transaction_features.max_deposit': 'max_deposit_value_1y',
    'company_transaction_features.max_withdrawal': 'max_withdrawal_value_1y',
    'company_transaction_features.pct_round_txns': 'percentage_of_round_transactions_1y',
    'company_transaction_features.tester_pmt_cnt': 'number_of_tester_payments_1y',
    'company_transaction_features.cashtxns_latehrs': 'number_of_latehours_cash_transactions_1y',
    'company_transaction_features.cardpmt_wtd_pct': 'percentage_cardpayments_of_all_withdrawals_1y',
    'company_transaction_features.fastpmt_wtd_pct': 'percentage_fastpayments_of_all_withdrawals_1y',
    'company_transaction_features.ddebit_wtd_pct': 'percentage_directdebits_of_all_withdrawals_1y',
    'company_transaction_features.cardwtd_wtd_pct': 'percentage_cardwithdrawals_of_all_withdrawals_1y',
    'company_transaction_features.outpmt_wtd_pct': 'percentage_outpayments_of_all_withdrawals_1y',
    'company_transaction_features.fastpmt_dep_pct': 'percentage_fastpayments_of_all_deposits_1y',
    'company_transaction_features.cash_dep_pct': 'percentage_cashdeposits_of_all_deposits_1y',
    'company_transaction_features.inpmt_dep_pct': 'percentage_inpayments_of_all_deposits_1y',
    'company_transaction_features.fastpmt_beneficiaries': 'number_of_fastpayment_beneficiaries_1y',
    'company_transaction_features.fastpmt_benefactors': 'number_of_fastpayment_benefactors_1y',
    'company_transaction_features.cardpmt_acceptors': 'number_of_card_acceptors_1y',
    'company_transaction_features.ddebit_beneficiaries': 'number_of_direct_debit_beneficiaries_1y',
    'company_transaction_features.deposit_wtd_frequency_ratio': 'deposit_withdrawal_frequency_ratio_1y',
    'company_transaction_features.hmrc_txns_cnt': 'number_of_hmrc_transactions_1y',
    'company_transaction_features.xero_txns_cnt': 'number_of_xero_transactions_1y',
    'company_transaction_features.pos_atm_locations': 'number_of_atms_used_1y',
    'company_transaction_features.card_pans_cnt': 'number_of_cards_used_1y',
    'company_transaction_features.high_card_pmts': 'number_of_high_card_payments_on_account_1y',
    'company_transaction_features.high_card_wtds': 'number_of_high_card_withdrawals_on_account_1y',
    'company_transaction_features.high_fpmt_out': 'number_of_high_outgoing_fastpayments_on_account_1y',
    'company_transaction_features.high_pmt_out': 'number_of_high_outpayments_on_account_1y',
    'company_transaction_features.high_ddebit': 'number_of_high_directdebits_on_account_1y',
    'company_transaction_features.high_fpmt_in': 'number_of_high_incoming_fastpayments_on_account_1y',
    'company_transaction_features.max_cash_deposits': 'number_of_maximum_cash_deposits_on_account_1y',
    'company_transaction_features.high_pmt_in': 'number_of_high_inpayments_on_account_1y',
    'rolling_sum_features.deposits_sum_2h': 'rolling_sum_of_deposits_on_account_2h',
    'rolling_sum_features.deposits_sum_4h': 'rolling_sum_of_deposits_on_account_4h',
    'rolling_sum_features.deposits_sum_8h': 'rolling_sum_of_deposits_on_account_8h',
    'rolling_sum_features.deposits_sum_12h': 'rolling_sum_of_deposits_on_account_12h',
    'rolling_sum_features.deposits_sum_24h': 'rolling_sum_of_deposits_on_account_24h',
    'rolling_sum_features.deposits_sum_48h': 'rolling_sum_of_deposits_on_account_48h',
    'rolling_sum_features.withdrawals_sum_2h': 'rolling_sum_of_withdrawals_on_account_2h',
    'rolling_sum_features.withdrawals_sum_4h': 'rolling_sum_of_withdrawals_on_account_4h',
    'rolling_sum_features.withdrawals_sum_8h': 'rolling_sum_of_withdrawals_on_account_8h',
    'rolling_sum_features.withdrawals_sum_12h': 'rolling_sum_of_withdrawals_on_account_12h',
    'rolling_sum_features.withdrawals_sum_24h': 'rolling_sum_of_withdrawals_on_account_24h',
    'rolling_sum_features.withdrawals_sum_48h': 'rolling_sum_of_withdrawals_on_account_48h',
    'rolling_sum_features.incoming_payments_sum_2h': 'rolling_sum_of_incoming_payments_on_account_2h',
    'rolling_sum_features.incoming_payments_sum_4h': 'rolling_sum_of_incoming_payments_on_account_4h',
    'rolling_sum_features.incoming_payments_sum_8h': 'rolling_sum_of_incoming_payments_on_account_8h',
    'rolling_sum_features.incoming_payments_sum_12h': 'rolling_sum_of_incoming_payments_on_account_12h',
    'rolling_sum_features.incoming_payments_sum_24h': 'rolling_sum_of_incoming_payments_on_account_24h',
    'rolling_sum_features.incoming_payments_sum_48h': 'rolling_sum_of_incoming_payments_on_account_48h',
    'rolling_sum_features.incoming_payments_sum_60days': 'incoming_payments_sum_account_60d',
    'rolling_sum_features.outgoing_payments_sum_2h': 'rolling_sum_of_outgoing_payments_on_account_2h',
    'rolling_sum_features.outgoing_payments_sum_4h': 'rolling_sum_of_outgoing_payments_on_account_4h',
    'rolling_sum_features.outgoing_payments_sum_8h': 'rolling_sum_of_outgoing_payments_on_account_8h',
    'rolling_sum_features.outgoing_payments_sum_12h': 'rolling_sum_of_outgoing_payments_on_account_12h',
    'rolling_sum_features.outgoing_payments_sum_24h': 'rolling_sum_of_outgoing_payments_on_account_24h',
    'rolling_sum_features.outgoing_payments_sum_48h': 'rolling_sum_of_outgoing_payments_on_account_48h',
    'rolling_sum_features.outgoing_payments_sum_60days': 'outgoing_payments_sum_account_60d',
    'rolling_sum_features.deposits_mean_60days': 'average_deposit_value_60d',
    'rolling_sum_features.deposits_max_60days': 'max_deposit_value_60d',
    'rolling_sum_features.withdrawals_mean_60days': 'average_withdrawal_value_60d',
    'rolling_sum_features.withdrawals_max_60days': 'max_withdrawal_value_60d',
    'rolling_sum_features.outgoing_payments_over_threshold_sum_60days': 'number_of_withdrawals_over_threshold_60d',
    'rolling_sum_features.last_payment_request_max_60days': 'last_payment_requested_received_at_60d',
    'rolling_sum_features.tester_payments_sum_24h': 'number_of_tester_payments_24h',
    'withdrawals_to_counter_party_features.withdrawals_to_cp_sum_2h': 'rolling_sum_of_withdrawals_to_counter_party_2h',
    'withdrawals_to_counter_party_features.withdrawals_to_cp_sum_48h': 'rolling_sum_of_withdrawals_to_counter_party_48h',
    'withdrawals_to_counter_party_features.withdrawals_sum_60days': 'rolling_sum_of_withdrawals_to_counter_party_60d',
    'withdrawals_to_counter_party_features.withdrawals_to_cp_mean_2h': 'rolling_mean_of_withdrawals_to_counter_party_2h',
    'withdrawals_to_counter_party_features.withdrawals_to_cp_mean_48h': 'rolling_mean_of_withdrawals_to_counter_party_48h',
    'withdrawals_to_counter_party_features.withdrawals_to_cp_mean_60days': 'rolling_mean_of_withdrawals_to_counter_party_60d',
    'withdrawals_to_counter_party_features.withdrawals_to_cp_max_2h': 'rolling_max_of_withdrawals_to_counter_party_2h',
    'withdrawals_to_counter_party_features.withdrawals_to_cp_max_48h': 'rolling_max_of_withdrawals_to_counter_party_48h',
    'withdrawals_to_counter_party_features.withdrawals_to_cp_max_60days': 'rolling_max_of_withdrawals_to_counter_party_60d',
    'withdrawals_to_counter_party_features.withdrawals_to_cp_min_2h': 'rolling_min_of_withdrawals_to_counter_party_2h',
    'withdrawals_to_counter_party_features.withdrawals_to_cp_min_48h': 'rolling_min_of_withdrawals_to_counter_party_48h',
    'withdrawals_to_counter_party_features.withdrawals_to_cp_min_60days': 'rolling_min_of_withdrawals_to_counter_party_60d',
    'withdrawals_to_counter_party_features.payments_to_cp_sum_2h': 'rolling_count_of_payments_to_counter_party_2h',
    'withdrawals_to_counter_party_features.payments_to_cp_sum_48h': 'rolling_count_of_payments_to_counter_party_48h',
    'withdrawals_to_counter_party_features.payments_to_cp_sum_60days': 'rolling_count_of_payments_to_counter_party_60d',
    'deposits_from_counter_party_features.deposits_from_cp_sum_2h': 'rolling_sum_of_deposits_from_counter_party_2h',
    'deposits_from_counter_party_features.deposits_from_cp_sum_48h': 'rolling_sum_of_deposits_from_counter_party_48h',
    'deposits_from_counter_party_features.deposits_from_cp_sum_60d': 'rolling_sum_of_deposits_from_counter_party_60d',
    'deposits_from_counter_party_features.deposits_from_cp_mean_2h': 'rolling_mean_of_deposits_from_counter_party_2h',
    'deposits_from_counter_party_features.deposits_from_cp_mean_48h': 'rolling_mean_of_deposits_from_counter_party_48h',
    'deposits_from_counter_party_features.deposits_from_cp_mean_60d': 'rolling_mean_of_deposits_from_counter_party_60d',
    'deposits_from_counter_party_features.deposits_from_cp_max_2h': 'rolling_max_of_deposits_from_counter_party_2h',
    'deposits_from_counter_party_features.deposits_from_cp_max_48h': 'rolling_max_of_deposits_from_counter_party_48h',
    'deposits_from_counter_party_features.deposits_from_cp_max_60d': 'rolling_max_of_deposits_from_counter_party_60d',
    'deposits_from_counter_party_features.deposits_from_cp_min_2h': 'rolling_min_of_deposits_from_counter_party_2h',
    'deposits_from_counter_party_features.deposits_from_cp_min_48h': 'rolling_min_of_deposits_from_counter_party_48h',
    'deposits_from_counter_party_features.deposits_from_cp_min_60d': 'rolling_min_of_deposits_from_counter_party_60d',
    'deposits_from_counter_party_features.payments_from_cp_sum_2h': 'rolling_count_of_payments_from_counter_party_2h',
    'deposits_from_counter_party_features.payments_from_cp_sum_48h': 'rolling_count_of_payments_from_counter_party_48h',
    'deposits_from_counter_party_features.payments_from_cp_sum_60days': 'rolling_count_of_payments_from_counter_party_60d',

    'deposits_from_counter_party_features.sum_payments_from_cp_over_threshold_1000_2h': 'rolling_sum_of_deposits_to_counter_party_abv_thresh_1000_2h',
    'deposits_from_counter_party_features.sum_payments_from_cp_over_threshold_1000_48h': 'rolling_sum_of_deposits_to_counter_party_abv_thresh_1000_48h',
    'deposits_from_counter_party_features.sum_payments_from_cp_over_threshold_1000_60days': 'rolling_sum_of_deposits_to_counter_party_abv_thresh_1000_60d',
    'deposits_from_counter_party_features.sum_payments_value_from_cp_over_threshold_1000_2h': 'rolling_sum_of_deposits_value_to_counter_party_abv_thresh_1000_2h',
    'deposits_from_counter_party_features.sum_payments_value_from_cp_over_threshold_1000_48h': 'rolling_sum_of_deposits_value_to_counter_party_abv_thresh_1000_48h',
    'deposits_from_counter_party_features.sum_payments_value_from_cp_over_threshold_1000_60days': 'rolling_sum_of_deposits_value_to_counter_party_abv_thresh_1000_60d',
    'withdrawals_to_counter_party_features.sum_payments_to_cp_over_threshold_1000_2h': 'rolling_sum_of_withdrawals_to_counter_party_abv_thresh_1000_2h',
    'withdrawals_to_counter_party_features.sum_payments_to_cp_over_threshold_1000_48h': 'rolling_sum_of_withdrawals_to_counter_party_abv_thresh_1000_48h',
    'withdrawals_to_counter_party_features.sum_payments_to_cp_over_threshold_1000_60days': 'rolling_sum_of_withdrawals_to_counter_party_abv_thresh_1000_60d',
    'withdrawals_to_counter_party_features.sum_payments_value_to_cp_over_threshold_1000_2h': 'rolling_sum_of_withdrawals_value_to_counter_party_abv_thresh_1000_2h',
    'withdrawals_to_counter_party_features.sum_payments_value_to_cp_over_threshold_1000_48h': 'rolling_sum_of_withdrawals_value_to_counter_party_abv_thresh_1000_48h',
    'withdrawals_to_counter_party_features.sum_payments_value_to_cp_over_threshold_1000_60days': 'rolling_sum_of_withdrawals_value_to_counter_party_abv_thresh_1000_60d',
    'withdrawals_to_counter_party_features.last_payment_request_timestamp_over_threshold_1000_60days': "last_pmt_timestamp_over_1000_60d",

    'payment_recipient_events_features_new_to_member.last_updated_at': 'timestamp_ntm_recipient_last_updated',
    'payment_recipient_events_features_new_to_member.confirmed_timestamp': 'timestamp_ntm_recipient_confirmed',
    'payment_recipient_events_features_new_to_tide.number_of_tide_accounts_with_same_payee': 'number_of_tide_accounts_with_the_same_payee',
    'payment_recipient_events_features_new_to_tide.first_time_payee_addition': 'timestamp_ntt_payee_first_added',
    'payment_recipient_events_features_new_to_tide.last_time_payee_addition': 'timestamp_ntt_payee_last_added',

    'recency_features.is_payment_initiated_between_9am_to_10pm_sum_60d': 'number_of_payments_outside_9am_to_10pm_60d',
    'recency_features.outgoing_payments_over_1000_sum_2d': 'number_of_high_outgoing_fastpayments_on_account_48h',
    'recency_features.outgoing_payments_over_1000_sum_60d': 'number_of_high_outgoing_fastpayments_on_account_60d',
    'recency_features.payment_initiated_hour_max_48h': 'payment_initiated_hour_max_48h',
    'recency_features.payment_initiated_hour_min_48h': 'payment_initiated_hour_min_48h',
    'recency_features.stddev_deposits_in_60d': 'stddev_deposits_in_60d',
    'recency_features.stddev_withdrawls_in_60d': 'stddev_withdrawals_in_60d',
    'payee_new_to_member_agg_features.new_payee_count_30mins': 'new_payee_count_30mins',
    'payee_new_to_member_agg_features.new_payee_count_2h': 'new_payee_count_2h',
    'payee_new_to_member_agg_features.new_payee_count_48h': 'new_payee_count_48h',
}

INPUT_FEATURES_PRE_TRANSFORMATION = list(FEATURE_MAPPER.values())

# This is to support feature check between model and tecton FS

MODEL_SIGNATURE_COLS = list(set(FEATURE_MAPPER.keys()) - {'is_registered', 'company_type', 'kyc_risk_band'})
# MODEL_SIGNATURE_COLS = INPUT_FEATURES_PRE_TRANSFORMATION.copy()
# [MODEL_SIGNATURE_COLS.remove(x) for x in ['is_registered_company', 'registered_company_type', 'rule_feature_kyc_risk_band']]
MODEL_SIGNATURE_COLS = MODEL_SIGNATURE_COLS + [
    'ifre_member_features.is_registered', 'company_core_features.is_registered',
    'ifre_member_features.company_type', 'company_core_features.company_type',
    'whitelisting_features.risk_band', 'company_core_features.risk_band',
]

# COMMAND ----------


APP_FEATURES = [
    'rolling_sum_of_withdrawals_to_counter_party_2h', 'rolling_sum_of_withdrawals_to_counter_party_48h',
    'rolling_sum_of_withdrawals_to_counter_party_60d', 'rolling_mean_of_withdrawals_to_counter_party_2h',
    'rolling_mean_of_withdrawals_to_counter_party_48h', 'rolling_mean_of_withdrawals_to_counter_party_60d',
    'rolling_max_of_withdrawals_to_counter_party_2h', 'rolling_max_of_withdrawals_to_counter_party_48h',
    'rolling_max_of_withdrawals_to_counter_party_60d', 'rolling_min_of_withdrawals_to_counter_party_2h',
    'rolling_min_of_withdrawals_to_counter_party_48h', 'rolling_min_of_withdrawals_to_counter_party_60d',
    'rolling_count_of_payments_to_counter_party_2h', 'rolling_count_of_payments_to_counter_party_48h',
    'rolling_count_of_payments_to_counter_party_60d', 'rolling_sum_of_deposits_from_counter_party_2h',
    'rolling_sum_of_deposits_from_counter_party_48h', 'rolling_sum_of_deposits_from_counter_party_60d',
    'rolling_mean_of_deposits_from_counter_party_2h', 'rolling_mean_of_deposits_from_counter_party_48h',
    'rolling_mean_of_deposits_from_counter_party_60d', 'rolling_max_of_deposits_from_counter_party_2h',
    'rolling_max_of_deposits_from_counter_party_48h', 'rolling_max_of_deposits_from_counter_party_60d',
    'rolling_min_of_deposits_from_counter_party_2h', 'rolling_min_of_deposits_from_counter_party_48h',
    'rolling_min_of_deposits_from_counter_party_60d', 'rolling_count_of_payments_from_counter_party_2h',
    'rolling_count_of_payments_from_counter_party_48h', 'rolling_count_of_payments_from_counter_party_60d',
    'number_of_tide_accounts_with_the_same_payee', 'rolling_sum_of_outgoing_payments_on_account_48h',
    'rolling_sum_of_withdrawals_on_account_2h', 'rolling_sum_of_deposits_on_account_2h',
    "number_of_payments_outside_9am_to_10pm_60d", "number_of_high_outgoing_fastpayments_on_account_48h",
    "number_of_high_outgoing_fastpayments_on_account_60d", "payment_initiated_hour_max_48h",
    "payment_initiated_hour_min_48h", 'stddev_deposits_in_60d', 'stddev_withdrawals_in_60d',
    'rolling_sum_of_deposits_to_counter_party_abv_thresh_1000_2h',
    'rolling_sum_of_deposits_to_counter_party_abv_thresh_1000_48h',
    'rolling_sum_of_deposits_to_counter_party_abv_thresh_1000_60d',
    'rolling_sum_of_deposits_value_to_counter_party_abv_thresh_1000_2h',
    'rolling_sum_of_deposits_value_to_counter_party_abv_thresh_1000_48h',
    'rolling_sum_of_deposits_value_to_counter_party_abv_thresh_1000_60d',
    'rolling_sum_of_withdrawals_to_counter_party_abv_thresh_1000_2h',
    'rolling_sum_of_withdrawals_to_counter_party_abv_thresh_1000_48h',
    'rolling_sum_of_withdrawals_to_counter_party_abv_thresh_1000_60d',
    'rolling_sum_of_withdrawals_value_to_counter_party_abv_thresh_1000_2h',
    'rolling_sum_of_withdrawals_value_to_counter_party_abv_thresh_1000_48h',
    'rolling_sum_of_withdrawals_value_to_counter_party_abv_thresh_1000_60d',
    'new_payee_count_30mins',
    'new_payee_count_2h',
    'new_payee_count_48h',
]

# COMMAND ----------


RULE_FEATURES = [
    'rule_feature_is_clearlisted', 'rule_feature_is_recipient_account_exempted',
    'rule_feature_is_matching_credit_event',
    'rule_feature_is_source_dest_clearlisted'
]
nhours = [2, 4, 8, 12, 24, 48]

FILLNA_COLS_WITH_ZERO = [f"rolling_sum_of_deposits_on_account_{x}h" for x in nhours] + [
    f"rolling_sum_of_withdrawals_on_account_{x}h" for x in nhours] + \
                        [f"rolling_sum_of_incoming_payments_on_account_{x}h" for x in nhours] + [
                            f"rolling_sum_of_outgoing_payments_on_account_{x}h" for x in nhours]

FILLNA_COLS_WITH_ONE = ["incoming_payments_sum_account_60d", "max_withdrawal_value_60d", "average_deposit_value_60d",
                        "max_deposit_value_60d", "stddev_deposits_in_60d", "stddev_withdrawals_in_60d",
                        "outgoing_payments_sum_account_60d"]

FILLNA_COLS_WITH_ZERO = FILLNA_COLS_WITH_ZERO + [
    'average_deposit_value_1y', 'max_deposit_value_1y', 'average_withdrawal_value_60d', 'average_withdrawal_value_1y',
    'max_withdrawal_value_1y', 'number_of_tester_payments_24h', 'number_of_withdrawals_over_threshold_60d',
    'percentage_cardpayments_of_all_withdrawals_1y', 'percentage_fastpayments_of_all_withdrawals_1y',
    'percentage_directdebits_of_all_withdrawals_1y', 'percentage_cardwithdrawals_of_all_withdrawals_1y',
    'percentage_outpayments_of_all_withdrawals_1y', 'percentage_fastpayments_of_all_deposits_1y',
    'percentage_cashdeposits_of_all_deposits_1y', 'percentage_inpayments_of_all_deposits_1y',
    'number_of_high_card_payments_on_account_1y', 'number_of_high_card_withdrawals_on_account_1y',
    'number_of_high_outgoing_fastpayments_on_account_1y', 'number_of_high_outpayments_on_account_1y',
    'number_of_high_directdebits_on_account_1y', 'number_of_high_incoming_fastpayments_on_account_1y',
    'number_of_maximum_cash_deposits_on_account_1y', 'number_of_high_inpayments_on_account_1y',
    'rolling_sum_of_withdrawals_to_counter_party_2h', 'rolling_sum_of_withdrawals_to_counter_party_48h',
    'rolling_sum_of_withdrawals_to_counter_party_60d', 'rolling_mean_of_withdrawals_to_counter_party_2h',
    'rolling_mean_of_withdrawals_to_counter_party_48h', 'rolling_mean_of_withdrawals_to_counter_party_60d',
    'rolling_max_of_withdrawals_to_counter_party_2h', 'rolling_max_of_withdrawals_to_counter_party_48h',
    'rolling_max_of_withdrawals_to_counter_party_60d', 'rolling_min_of_withdrawals_to_counter_party_2h',
    'rolling_min_of_withdrawals_to_counter_party_48h',
    'rolling_min_of_withdrawals_to_counter_party_60d', 'rolling_count_of_payments_to_counter_party_2h',
    'rolling_count_of_payments_to_counter_party_48h', 'rolling_count_of_payments_to_counter_party_60d',
    'rolling_sum_of_deposits_from_counter_party_2h', 'rolling_sum_of_deposits_from_counter_party_48h',
    'rolling_sum_of_deposits_from_counter_party_60d', 'rolling_mean_of_deposits_from_counter_party_2h',
    'rolling_mean_of_deposits_from_counter_party_48h', 'rolling_mean_of_deposits_from_counter_party_60d',
    'rolling_max_of_deposits_from_counter_party_2h', 'rolling_max_of_deposits_from_counter_party_48h',
    'rolling_max_of_deposits_from_counter_party_60d',
    'rolling_min_of_deposits_from_counter_party_2h', 'rolling_min_of_deposits_from_counter_party_48h',
    'rolling_min_of_deposits_from_counter_party_60d',
    'rolling_count_of_payments_from_counter_party_2h', 'rolling_count_of_payments_from_counter_party_48h',
    'rolling_count_of_payments_from_counter_party_60d',
    'number_of_tide_accounts_with_the_same_payee', 'rolling_sum_of_outgoing_payments_on_account_48h',
    'rolling_sum_of_withdrawals_on_account_2h', 'rolling_sum_of_deposits_on_account_2h',
    "number_of_payments_outside_9am_to_10pm_60d", "number_of_high_outgoing_fastpayments_on_account_48h",
    "number_of_high_outgoing_fastpayments_on_account_60d", "payment_initiated_hour_max_48h",
    "payment_initiated_hour_min_48h",
    "rolling_sum_of_deposits_to_counter_party_abv_thresh_1000_2h",
    "rolling_sum_of_deposits_to_counter_party_abv_thresh_1000_48h",
    "rolling_sum_of_deposits_to_counter_party_abv_thresh_1000_60d",
    "rolling_sum_of_deposits_value_to_counter_party_abv_thresh_1000_2h",
    'rolling_sum_of_deposits_value_to_counter_party_abv_thresh_1000_48h',
    'rolling_sum_of_deposits_value_to_counter_party_abv_thresh_1000_60d',
    'rolling_sum_of_withdrawals_to_counter_party_abv_thresh_1000_2h',
    'rolling_sum_of_withdrawals_to_counter_party_abv_thresh_1000_48h',
    'rolling_sum_of_withdrawals_to_counter_party_abv_thresh_1000_60d',
    'rolling_sum_of_withdrawals_value_to_counter_party_abv_thresh_1000_2h',
    'rolling_sum_of_withdrawals_value_to_counter_party_abv_thresh_1000_48h',
    'rolling_sum_of_withdrawals_value_to_counter_party_abv_thresh_1000_60d',
    'new_payee_count_30mins',
    'new_payee_count_2h',
    'new_payee_count_48h',
]

NUMERICAL_COLS = [
    'requested_payment_value', 'is_registered_company', 'member_age_at_onboarding', 'net_balance_indicator',
    'cashflow_value_indicator', 'deposit_withdrawal_ratio', 'cashflow_volume_indicator',
    'payment_requested_mean_withdrawal_ratio_60d', 'payment_requested_max_withdrawal_ratio_60d',
    'payment_requested_mean_deposit_ratio_60d', 'payment_requested_max_deposit_ratio_60d',
    'payment_requested_max_withdrawal_ratio_1y', 'payment_requested_max_deposit_ratio_1y',
    'payment_requested_average_deposit_ratio_1y', 'payment_requested_average_withdrawal_ratio_1y',
    'number_of_high_value_transactions_1y', 'number_of_withdrawals_over_threshold_60d',
    'number_of_days_since_last_withdrawal_60d', 'is_previous_payment_in_last_15mins',
    'fastpmt_beneficiaries_benefactors_ratio_1y', 'number_of_tester_payments_1y',
    'number_of_latehours_cash_transactions_1y',
    'number_of_card_acceptors_1y', 'number_of_direct_debit_beneficiaries_1y', 'number_of_hmrc_transactions_1y',
    'number_of_xero_transactions_1y', 'number_of_atms_used_1y',
    'number_of_cards_used_1y', 'number_of_tester_payments_24h', 'percentage_of_round_transactions_1y',
    'deposit_withdrawal_frequency_ratio_1y',
]

CATEGORICAL_COLS = [
    'member_area_code', 'member_id_country_issue', 'member_id_type', 'registered_company_type',
    'registered_company_industry_classification', 'payment_channel_most_used_for_withdrawals',
    'payment_channel_most_used_for_deposits',
]

NEW_ADDED = [
    'time_passed_since_confirmation_of_payee', 'payments_initiated_hour_48h_range', 'ratio_pmts_outside_business_60d',
    'incoming_payment_requested_avg_pmt_value_with_payee_ratio_60d',
    'outgoing_payment_requested_avg_pmt_value_with_payee_ratio_60d', 'mean_of_payments_to_new_payees_60d',
    'sum_of_payments_to_new_payees_48h', 'ratio_of_pmts_value_to_payee_of_overall_pmts_60d',
    'ratio_of_pmts_value_to_payee_of_overall_pmts_48h', 'ratio_of_pmts_value_to_payee_of_overall_pmts_2h',
    'rolling_sum_of_pmts_over_threshold_to_new_payee_48h', 'rolling_sum_of_pmts_over_threshold_to_new_payee_60d',
    'hours_since_confirmation_of_payee', 'days_since_confirmation_of_payee',
    'incoming_payment_density_over_the_past_2h_vs_60days', "incoming_payment_density_over_the_past_2h_vs_60days_ratio",
    'ratio_of_rolling_sum_of_wtd_2h_to_max_wtd_60d', 'ratio_of_rolling_sum_of_deposits_2h_to_max_deposit_60d',
    'count_above_thr_deposite_from_counter_party_1000_60d', 'payment_transfer_rate_on_cop',
    'requested_payment_to_avg_deposit_60d_ratio', 'is_requested_payment_gt_60day_max_wtd'
]

MODEL_INPUT_FEATURES = ['requested_payment_value', 'is_registered_company', 'member_age_at_onboarding',
                        'member_area_code', 'member_id_country_issue', 'member_id_type', 'registered_company_type',
                        'registered_company_industry_classification',
                        'net_balance_indicator', 'cashflow_value_indicator', 'deposit_withdrawal_ratio',
                        'cashflow_volume_indicator', 'payment_requested_mean_withdrawal_ratio_60d',
                        'payment_requested_max_withdrawal_ratio_60d',
                        'payment_requested_mean_deposit_ratio_60d', 'payment_requested_max_deposit_ratio_60d',
                        'payment_requested_max_withdrawal_ratio_1y', 'payment_requested_max_deposit_ratio_1y',
                        'payment_requested_average_deposit_ratio_1y',
                        'payment_requested_average_withdrawal_ratio_1y', 'payment_channel_most_used_for_withdrawals',
                        'payment_channel_most_used_for_deposits', 'number_of_high_value_transactions_1y',
                        'number_of_withdrawals_over_threshold_60d',
                        'number_of_days_since_last_withdrawal_60d', 'is_previous_payment_in_last_15mins',
                        'fastpmt_beneficiaries_benefactors_ratio_1y', 'number_of_tester_payments_1y',
                        'number_of_latehours_cash_transactions_1y',
                        'number_of_card_acceptors_1y', 'number_of_direct_debit_beneficiaries_1y',
                        'number_of_hmrc_transactions_1y', 'number_of_xero_transactions_1y', 'number_of_atms_used_1y',
                        'number_of_cards_used_1y', 'number_of_tester_payments_24h',
                        'percentage_of_round_transactions_1y', 'deposit_withdrawal_frequency_ratio_1y',
                        'outgoing_payments_sum_account_60d', 'incoming_payments_sum_account_60d'
                        ] + APP_FEATURES + NEW_ADDED

ONE_YEAR_FEATS = ['payment_requested_max_withdrawal_ratio_1y', 'payment_requested_max_deposit_ratio_1y',
                  'payment_requested_average_deposit_ratio_1y',
                  'payment_requested_average_withdrawal_ratio_1y', 'number_of_high_value_transactions_1y',
                  'fastpmt_beneficiaries_benefactors_ratio_1y', 'number_of_tester_payments_1y',
                  'number_of_latehours_cash_transactions_1y', 'percentage_of_round_transactions_1y',
                  'deposit_withdrawal_frequency_ratio_1y',
                  'number_of_card_acceptors_1y', 'number_of_direct_debit_beneficiaries_1y',
                  'number_of_hmrc_transactions_1y', 'number_of_xero_transactions_1y', 'number_of_atms_used_1y',
                  'number_of_cards_used_1y', 'payment_channel_most_used_for_withdrawals','payment_channel_most_used_for_deposits']

CATEGORICAL_COLS = [
    'member_area_code', 'member_id_country_issue', 'member_id_type', 'registered_company_type',
    'registered_company_industry_classification', 'payment_channel_most_used_for_withdrawals',
    'payment_channel_most_used_for_deposits',
]

FEATURES_TO_EXCLUDE = (
    set(ONE_YEAR_FEATS)
    .union({'trend_of_last_20_deposits',
    'trend_of_last_20_withdrawals',
    'zscore_last_20_withdrawals_on_account',
    'payment_outlier_zscore_last_20_withdrawals',
    'payment_outlier_zscore_last_20_deposits',
    'velocity_of_last_20_withdrawals_on_account',
    'velocity_of_last_20_deposits_on_account',
    'zscore_last_20_deposits_on_account',
    'avg_time_diff_between_last_20_txns_on_acct',})
)


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        # print("Finished {} in {} secs".format(repr(func.__name__), round(run_time, 5))) # Uncomment for time elapsed by method in feature transformation
        return value

    return wrapper
