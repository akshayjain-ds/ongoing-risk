# Databricks notebook source
# MAGIC %run ./set_up

# COMMAND ----------

# MAGIC %run ./train_metadata

# COMMAND ----------

day_of_month

# COMMAND ----------

# MAGIC %run ./pre-processing

# COMMAND ----------

# MAGIC %run ./functions

# COMMAND ----------

# MAGIC %run ./queries

# COMMAND ----------

# MAGIC %run ./utils_2

# COMMAND ----------

import pandas as pd
import json
import tecton
from optbinning import OptimalBinning
import gc
import category_encoders as ce
from datetime import datetime, timedelta

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


def spark_connector(query_string: str) -> pd.DataFrame:
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

training_df = pd.read_csv(f"/Workspace/Shared/Decisioning/Strawberries/ongoing-ntt_mp_risk-v1/artefacts/{'ntt_ftt' if include_ftt else 'ntt'}_{day_of_month}_base_{train_start_date}_{val_end_date}.csv.gz",
                          dtype={id1: str, id2: str})

training_df = training_df.assign(

  company_created_on = lambda col: pd.to_datetime(col[created_on]),
  timestamp = lambda col: pd.to_datetime(col[timestamp]),
  approved_at = lambda col:  pd.to_datetime(col['approved_at'])

)           
training_df.shape

# COMMAND ----------

training_df.head()

# COMMAND ----------

def get_data_from_fs(dataset_spine=None, timestamp_key=timestamp, feature_service=None, tecton_workspace="prod"):

  tecton_workspace = tecton.get_workspace(tecton_workspace)
  f_service = tecton_workspace.get_feature_service(feature_service)
  features = f_service.get_historical_features(dataset_spine, timestamp_key=timestamp_key).to_pandas()
  features.columns = [col.split("__")[1].lower() if col.__contains__("__") else col.lower() for col in features.columns]

  return features

def get_data_from_ds(dataset_spine=None, id_key=None, timestamp_key=created_on, data_service=None, tecton_workspace="prod"):

  tecton_workspace = tecton.get_workspace(tecton_workspace)
  d_service = tecton_workspace.get_data_source(data_service)
  data = d_service.get_dataframe(start_time=dataset_spine[timestamp_key].min(), 
                                 end_time=dataset_spine[timestamp_key].max()).to_pandas()
  data[created_on] = pd.to_datetime(data[created_on])
  data = dataset_spine.merge(data, how='left', on=[id_key, timestamp_key])
  data.columns = [col.split("__")[1].lower() if col.__contains__("__") else col.lower() for col in data.columns]

  return data

# COMMAND ----------

ongoing_features = get_data_from_fs(training_df.loc[:, [id1, timestamp]].drop_duplicates(),
                            timestamp,
                            ongoing_feature_service,
                            "prod")
ongoing_features.shape

# COMMAND ----------

features_to_test_df= spark_connector(
  features_to_test_query.format(
    from_date=str(training_df['approved_at'].min().date() - timedelta(days=30)),
    to_date=str(training_df['approved_at'].max().date() + timedelta(days=30)))
)
features_to_test_df = features_to_test_df.toPandas()
features_to_test_df.columns = [col.split("__")[1].lower() if col.__contains__("__") else col.lower() for col in features_to_test_df.columns]
features_to_test_df.shape

# COMMAND ----------

ongoing_features = ongoing_features.merge(features_to_test_df, on=[id1], how='inner')
ongoing_features.shape

# COMMAND ----------

ongoing_features = ongoing_features.assign(

  company_age_at_timestamp = lambda col: (pd.to_datetime(col[timestamp]).apply(lambda x : x.date()) - pd.to_datetime(col['incorporation_at']).apply(lambda x : x.date()))/np.timedelta64(1, 'M'),

  attribution_marketing_campaign = lambda col: col['last_touch_attribution_marketing_campaign_signup'].fillna("Unattributed")
  .apply(lambda x: x.lower().strip()),
  
  attribution_marketing_channel = lambda col: col['last_touch_attribution_marketing_channel_signup'].fillna("Unattributed")
  .apply(lambda x: x.lower().strip()),

  is_cofo = lambda col: col['is_cofo'].fillna(0),

  receipt_uploaded_before_timestamp = lambda col: np.where(pd.to_datetime(col['first_receipt_uploaded_at']) < col[timestamp], 1, 0),

  receipt_match_before_timestamp = lambda col: np.where(pd.to_datetime(col['first_receipt_match_at']) < col[timestamp], 1, 0),

  first_invoice_before_timestamp = lambda col: np.where(pd.to_datetime(col['date_of_first_invoice']) < col[timestamp], 1, 0),

  invoice_matched_before_timestamp = lambda col: np.where(pd.to_datetime(col['first_invoice_matched_at']) < col[timestamp], 1, 0),

  invoice_chased_before_timestamp = lambda col: np.where(pd.to_datetime(col['first_invoice_chased_at']) < col[timestamp], 1, 0),
  
  activity_before_timestamp = lambda col: np.where(pd.to_datetime(col['last_activity_at']) < col[timestamp], 1, 0),

  login_before_timestamp = lambda col: np.where(pd.to_datetime(col['last_login_at']) < col[timestamp], 1, 0),
  
)
ongoing_features.shape

# COMMAND ----------

ongoing_features.groupby([timestamp])[[f'company_age_at_{timestamp}']].mean()

# COMMAND ----------

ongoing_features.drop(columns= [
  'industry_classification', 
  'applicant_postcode', 
  'applicant_id_type',
  'age_at_completion',
  'applicant_id_country_issue',
  'company_type',
  'is_registered',
  'mkyc_ind',
  'risk_band',
  'incorporation_at',
  'last_touch_attribution_marketing_campaign_signup',
  'last_touch_attribution_marketing_channel_signup', 
  'is_cofo',
  'first_receipt_uploaded_at', 
  'first_receipt_match_at',
  'date_of_first_invoice', 
  'first_invoice_matched_at',
  'first_invoice_chased_at',
  'last_activity_at',
  'last_login_at'], inplace=True)

# COMMAND ----------

pd.isnull(ongoing_features).sum()/ongoing_features.shape[0]

# COMMAND ----------

ongoing_features.head()

# COMMAND ----------

del features_to_test_df
gc.collect()

# COMMAND ----------

ongoing_features.to_csv(f"/Workspace/Shared/Decisioning/Strawberries/ongoing-ntt_mp_risk-v1/artefacts/{'ntt_ftt' if include_ftt else 'ntt'}_{day_of_month}_raw_features_{train_start_date}_{val_end_date}.csv.gz", index=False, compression='gzip')

# COMMAND ----------

onboarding_features = get_data_from_ds(dataset_spine=training_df.loc[:, [id1, 
                                                                         created_on, 
                                                                         'approved_at']].drop_duplicates(),
                                      id_key = id1,
                                      timestamp_key=created_on,
                                      data_service=onboarding_data_source,
                                      tecton_workspace='akshayjain')
onboarding_features.shape

# COMMAND ----------

onboarding_features.head()

# COMMAND ----------

pd.isnull(onboarding_features).sum()/onboarding_features.shape[0]

# COMMAND ----------

onboarding_features.head()

# COMMAND ----------

duedil_df = spark_connector(duedil_query.format(from_date=str(training_df[created_on].min().date()),
                                              to_date=str(training_df[created_on].max().date())))
duedil_df = duedil_df.toPandas()
duedil_df.columns = [col.split("__")[1].lower() if col.__contains__("__") else col.lower() for col in duedil_df.columns]
duedil_df.drop(columns=['created_at'], inplace=True)
duedil_df.shape

# COMMAND ----------

duedil_df.head()

# COMMAND ----------

onboarding_features = onboarding_features.merge(duedil_df, on=[id1], how='left')
onboarding_features.shape

# COMMAND ----------

onboarding_features.head()

# COMMAND ----------

onboarding_features['duedil_hit'].fillna(0).mean()

# COMMAND ----------

onboarding_features = onboarding_features.assign(

  directors_info = lambda col: col['directorstree']
  .apply(lambda x: get_director_info(x)),

  applicant_director_nationality = lambda col: col[['applicant_id_firstname_rawdata',
                                                      'applicant_id_lastname_rawdata',
                                                      'applicant_dob_rawdata',
                                                      'directors_info']]
  .apply(lambda row: get_applicant_director_nationality(row), axis=1)
  .apply(lambda x: x if isinstance(x, list) else [x]),

  applicant_nationality_rawdata = lambda col: col['applicant_director_nationality'] + col['applicant_nationality_rawdata'].apply(lambda x: [x] if isinstance(x, str) else []),

  applicant_nationality_list = lambda col: col['applicant_nationality_rawdata']
  .apply(lambda x: np.unique(list(filter(lambda x: x not in [None, float('nan'), 'UNKNOWN'], x)))),

  applicant_nationality = lambda col: col['applicant_nationality_list']
  .apply(lambda countries: [validate_and_get_country(x) for x in countries]),

  company_postcode = lambda col: col['company_postcode_rawdata']
  .apply(lambda x: validate_and_get_postcode(x))
  .apply(lambda x: "".join(filter(lambda y: not y.isdigit(), x)))
  .apply(lambda x: np.NaN if str(x).__contains__("Error") else x)
  .apply(lambda x: x.upper() if isinstance(x, str) else x),

  company_structurelevelwise_1 = lambda col: col['structurelevelwise']
  .apply(lambda x: json.loads(x) if isinstance(x, str) else {})
  .apply(lambda x: x.get("1", np.NaN))
  .apply(lambda x: "1" if x == 1 else ("2" if x == 2 else ("3+" if x >= 3 else np.NaN))),

  company_directors_count = lambda col: col['directors_info']
  .apply(lambda x: len(x) if bool(x) else np.nan)
  .apply(lambda x: "1" if x == 1 else ("2" if x == 2 else ("3+" if x >= 3 else np.NaN)))

)

nationality_count = onboarding_features['applicant_nationality'].apply(lambda x: len(x)).max()
onboarding_features = onboarding_features.merge(onboarding_features[['applicant_nationality']].apply(lambda row: {f"applicant_nationality_{i}": j for i, j in enumerate(row['applicant_nationality'])}, axis=1, result_type='expand'), left_index=True, right_index=True)

onboarding_features.shape

# COMMAND ----------

onboarding_features = onboarding_features.assign(
  
  applicant_postcode = lambda col: col['applicant_postcode_rawdata']
  .apply(lambda x: validate_and_get_postcode(x))
  .apply(lambda x: "".join(filter(lambda y: not y.isdigit(), x))).apply(lambda x: np.NaN if str(x).__contains__("Error") else x)
  .apply(lambda x: str(x).upper()),

  applicant_idcountry_issue = lambda col: col['applicant_idcountry_issue_rawdata']
  .apply(lambda x: validate_and_get_country(x)).apply(lambda x: np.NaN if str(x).__contains__("Error") else x),

  applicant_id_type = lambda col: col[['applicant_id_type_rawdata',
                                       'applicant_id_subtype_rawdata',
                                       'applicant_idcountry_issue_rawdata']]
  .apply(lambda row: applicant_id_value_mapper(
    id_type=row['applicant_id_type_rawdata'],
    id_subtype=row['applicant_id_subtype_rawdata'],
    id_country=row['applicant_idcountry_issue_rawdata'],
    ), axis=1),
  
  company_sic = lambda col: col['company_sic_rawdata']
  .apply(lambda x: validate_and_get_company_sic(x))
  .apply(lambda x: [] if str(x).__contains__("Error") or str(x).__contains__("[]") else x),

  applicant_email_numeric = lambda col: col['applicant_email_rawdata']
  .astype(str)
  .apply(lambda x: any(re.findall(r'\d+', x)))*1,
  
  applicant_email_domain = lambda col: col['applicant_email_rawdata']
  .astype(str)
  .apply(lambda x: x.split("@")[-1]).apply(lambda x: x.split("#")[-1])
  .apply(lambda x: x if x in [
    'gmail.com',
    'hotmail.com',
    'outlook.com',
    'yahoo.com',
    'hotmail.co.uk',
    'icloud.com',
    'yahoo.co.uk',
    'live.co.uk'] else "other"),
  
  applicant_age_at_completion = lambda col: ((pd.to_datetime(col[created_on]).apply(lambda x : x.date()) - pd.to_datetime(col['applicant_dob_rawdata']).apply(lambda x : x.date()))/np.timedelta64(1, 'Y'))
  .apply(lambda x: np.NaN if np.around(x, 0) < 18  or np.around(x, 0) >100 else x),
  
  days_to_approval = lambda col: (pd.to_datetime(col['approved_at']).apply(lambda x : x.date()) - pd.to_datetime(col[created_on]).apply(lambda x : x.date()))/np.timedelta64(1, 'D'),
  
  applicant_device_type = lambda col: col['applicant_device_type_rawdata']
  .apply(lambda x: x.lower().strip() if isinstance(x, str) else np.NaN),
  
  company_icc = lambda col: col['company_icc_rawdata']
  .apply(lambda x: x.lower().strip() if isinstance(x, str) else np.NaN),
  
  applicant_years_to_id_expiry = lambda col: ((pd.to_datetime(col['applicant_id_expiry_rawdata']).apply(lambda x : x.date()) - pd.to_datetime(col[created_on]).apply(lambda x : x.date()))/np.timedelta64(1, 'Y'))
  .apply(lambda x: np.NaN if x < 0  or np.around(x, 0) > 10 else x),
  
  company_type = lambda col: col['company_type_rawdata']
  .apply(lambda x: "SOLE-TRADER" if x in [None, "NONE", "None", np.NaN, '', 'nan'] else ("LTD" if str(x).upper() == "LTD" else "other")),

  is_restricted_keyword_present = lambda col: col['company_trading_name_rawdata']
  .apply(lambda x: get_keywords(str(x).lower().strip())[0])*1,
  
  manual_approval_triggers = lambda col: col['manual_approval_triggers_rawdata']
  .apply(lambda x: x.split(',') if isinstance(x, str) else []),
  
  company_is_registered = lambda col: col['company_is_registered_rawdata'].fillna(0)
  
)

sic_count = onboarding_features['company_sic'].apply(lambda x: len(x)).max()
onboarding_features = onboarding_features.merge(onboarding_features[['company_sic']].apply(lambda row: {f"company_sic_{i}": j for i, j in enumerate(row['company_sic'])}, axis=1, result_type='expand'), left_index=True, right_index=True)

onboarding_features.shape

# COMMAND ----------

onboarding_features.drop(columns=['applicant_id_type_rawdata',
                                   'applicant_id_subtype_rawdata',
                                   'applicant_place_of_birth_rawdata',
                                   'applicant_fraud_status_rawdata',
                                   'director_fraud_status_rawdata',
                                   'shareholder_fraud_status_rawdata',
                                   'applicant_postcode_rawdata',
                                   'applicant_idcountry_issue_rawdata',
                                   'applicant_nationality_rawdata',
                                   'company_sic_rawdata',
                                   'company_sic',
                                  #  'company_keywords',
                                   'company_trading_name_rawdata',
                                   'applicant_email_rawdata',
                                   'applicant_dob_rawdata',
                                   'applicant_device_type_rawdata',
                                   'company_icc_rawdata',
                                   'applicant_id_firstname_rawdata',
                                   'applicant_id_lastname_rawdata',
                                   'applicant_id_expiry_rawdata',
                                   'company_type_rawdata',
                                   'manual_approval_triggers_rawdata',
                                   'company_is_registered_rawdata',
                                   'company_incorporation_date_rawdata',
                                   'approved_at',
                                   'duedil_created_at', 
                                   'duedil_hit',
                                   'companies_house_number',
                                   'company_countrycode', 
                                   'charitableidentitycount', 
                                   'financialsummary',
                                   'incorporationdate', 
                                   'numberofemployees', 
                                   'recentstatementdate',
                                   'majorshareholders', 
                                   'directorstree', 
                                   'shareholdertree',
                                   'personsofsignificantcontrol', 
                                   'structuredepth', 
                                   'structurelevelwise',
                                   'status', 
                                   'rnk',
                                   'company_postcode_rawdata',
                                   'directors_info',
                                   'applicant_director_nationality',
                                   'applicant_nationality_rawdata',
                                   'applicant_nationality_list',
                                   'applicant_nationality',
                                   timestamp,
                                   'duedil_payload'], inplace=True)
gc.collect()

# COMMAND ----------

pd.isnull(onboarding_features).sum()/onboarding_features.shape[0]

# COMMAND ----------

onboarding_features.dropna(subset=['applicant_age_at_completion'], inplace=True)
onboarding_features.shape

# COMMAND ----------

del duedil_df
gc.collect()

# COMMAND ----------

onboarding_features.head()

# COMMAND ----------

onboarding_features.to_csv(f"/Workspace/Shared/Decisioning/Strawberries/ongoing-ntt_mp_risk-v1/artefacts/{'ntt_ftt' if include_ftt else 'ntt'}_{day_of_month}_raw_features_{train_start_date}_{val_end_date}.csv.gz", index=False, compression='gzip')

# COMMAND ----------

features = ongoing_features.merge(onboarding_features, how='inner', on=[id1])
features.shape

# COMMAND ----------

features.head()

# COMMAND ----------

features = training_df[['company_id', timestamp, 'approved_at', 
                        target_b, target_d, target_c, 
                        'days_on_books', 'is_ntt', 'is_ftt', 'days_to_transact', 
                        f"days_remaining_as_{'ntt_ftt' if include_ftt else 'ntt'}",
                        ]].merge(features, on=[id1, timestamp])
features.shape

# COMMAND ----------

pd.isnull(features).sum()/features.shape[0]

# COMMAND ----------

features.info()

# COMMAND ----------

features.columns.tolist()

# COMMAND ----------

features.to_csv(f"/Workspace/Shared/Decisioning/Strawberries/ongoing-ntt_mp_risk-v1/artefacts/{'ntt_ftt' if include_ftt else 'ntt'}_{day_of_month}_raw_features_{train_start_date}_{val_end_date}.csv.gz", index=False, compression='gzip')
