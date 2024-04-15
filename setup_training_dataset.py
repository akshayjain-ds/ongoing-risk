# Databricks notebook source
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("dbfs_path_prefix", "dbfs:/tmp/ofre-retraining-2023/DD-4462/single_day/")
dbutils.widgets.dropdown("startdate_year", "2018", [str(x) for x in range(2018, 2024)])
dbutils.widgets.dropdown("startdate_month", "1", [str(x) for x in range(1, 13)])
dbutils.widgets.dropdown("stopdate_year", "2018", [str(x) for x in range(2018, 2024)])
dbutils.widgets.dropdown("stopdate_month", "1", [str(x) for x in range(1, 13)])
dbutils.widgets.text("multiple_days", "False")
dbutils.widgets.text("feature_service_name", "ongoing_risk_feature_service:v3")

dbfs_path_prefix = dbutils.widgets.getArgument(name='dbfs_path_prefix')
multiple_days = dbutils.widgets.getArgument(name='multiple_days')
startdate_year = int(dbutils.widgets.getArgument(name='startdate_year'))
startdate_month = int(dbutils.widgets.getArgument(name='startdate_month'))
stopdate_year = int(dbutils.widgets.getArgument(name='stopdate_year'))
stopdate_month = int(dbutils.widgets.getArgument(name='stopdate_month'))
import datetime

assert datetime.date(startdate_year, startdate_month, 1) >= datetime.date(2018, 6, 1)
if startdate_year != stopdate_year and startdate_month != stopdate_month:
    assert datetime.date(startdate_year, startdate_month, 1) < datetime.date(stopdate_year, stopdate_month, 1)

print(dbfs_path_prefix, multiple_days)

# COMMAND ----------

import os
import pyspark.sql.functions as F
from pyspark.sql.functions import col
import numpy as np
from pyspark.sql.types import StringType

logging.getLogger('py4j').setLevel(logging.ERROR)


def setup_active_companies_spine(feature_date: str) -> DataFrame:
    """Returns spark dataframe with spine representing companies for ongoing risk

    Args:
      feature_date (str): Date on which features are calculated

    Returns:
      pyspark.sql.DataFrame: List of active companies as on feature date
    """
    spine_df = spark_connector(GET_MEMBERS_QUERY.format(FEATURE_DATE=feature_date))
    spine_df = spine_df.withColumnRenamed("COMPANY_ID", "company_id")  # Rename company-id column
    spine_df = spine_df.withColumn("company_id", col("company_id").cast(StringType()))
    spine_df = spine_df.withColumn("timestamp", to_timestamp("timestamp"))  # Casting timestamp column
    return spine_df


def setup_training_dataset_spine(start_year: int, start_month: int, end_year: int, end_month: int, multiple_day) -> DataFrame:
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
    if start_year == end_year and start_month == end_month:  # if spine requested for only one month
        feature_date = get_random_day_of_month(start_year, start_month).strftime(
            '%Y-%m-%d')  # Choose random day of month
        spine_df = setup_active_companies_spine(feature_date)  # get active companies to include in spine
    else:  # if spine requested for a time period i.e. multiple months
        feature_dates = get_randomized_dates(start_year, start_month, end_year, end_month,
                                             multiple_day)  # Get random dates for every month in entire time period
        spines_list = [setup_active_companies_spine(each_date) for each_date in
                       feature_dates]  # Get active companies on each of the date
        spine_df = functools.reduce(DataFrame.union,
                                    spines_list)  # Union all spark dataframes to make consolidate spine
    return spine_df


def save_dataset(output_df: DataFrame, dataset_name: str):
    """Returns a deterministic set of randomized dates from each month in the time period specified

    Args:
      start_year (int): Timeperiod starting year
      start_month (int): Timeperiod starting month
      end_year (int): Timeperiod ending year
      end_month (int): Timeperiod ending month

    Returns:
      pyspark.sql.DataFrame: List of active companies to be included in the spine for the entire training period
    """
    import os
    outname = dataset_name + ".csv"
    outdir = dbfs_path_prefix.replace("dbfs:", "/dbfs")
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    fullname = os.path.join(outdir, outname)
    output_df.to_csv(fullname)


# COMMAND ----------

from datetime import timedelta


def get_data_from_fs(dataset_spine=None, timestamp_key="timestamp", feature_service=None, tecton_workspace="prod"):
    """
    Returns a pandas dataframe with historical features for the requested timestamp
      Args:
        dataset_spine (pyspark.sql.DataFrame): Timestamps for which features are needed
        timestamp_key (str): name of the column to be considered as timestamp by tecton
        feature_service (str): Feature service in tecton from which features are needed
        tecton_workspace (str): Tecton workspace
      Returns:
        pd.DataFrame: Set of historical features for requested timestamp

    """
    if dataset_spine and feature_service:
        tecton_workspace = tecton.get_workspace(tecton_workspace)
        service = tecton_workspace.get_feature_service(feature_service)
        features = service.get_historical_features(dataset_spine, timestamp_key=timestamp_key).to_pandas()
        window_end = features["timestamp"] + timedelta(days=90)
        features["window_end"] = window_end
        return features


def get_sar_labels(dataset_spine=None, feature_df=None):
    """
    Returns a pandas dataframe with SAR labels for the requested comapny_ids at requested timestamps
      Args:
        feature_df (pd.DataFrame): Historical features obtained from Tecton
      Returns:
        pd.DataFrame: Returns a pandas dataframe with SAR labels obtained from SQL for the requested comapny_ids at requested timestamps
    """

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
    """Processes the obtained SAR labels from SQL to get SAR duration for companies
      Args:
        fincrime_records_set (pd.DataFrame):  pandas dataframe with SAR labels obtained from SQL
      Returns:
        pd.DataFrame: Returns 'fincrime90_duration' feature for all company_ids
    """

    final_df = pd.DataFrame()
    for df_1 in fincrime_records_set:
        df_1['timestamp'] = df_1['timestamp'].apply(pd.Timestamp)
        df_1["duration"] = df_1['FIRST_SAR_CREATED_DT'].astype('datetime64[ns]', copy=False) - df_1['timestamp'].astype(
            'datetime64[ns]', copy=False)
        df_1["duration"] = df_1["duration"].dt.days
        df_2 = df_1[["duration", "COMPANY_ID", "timestamp", 'FIRST_SAR_CREATED_DT']]
        df_2.rename(columns={'COMPANY_ID': 'company_id', 'FIRST_SAR_CREATED_DT': 'first_sar_created_dt'}, inplace=True)
        df_2.drop_duplicates(subset=['duration', 'company_id'], keep="first", inplace=True)
        final_df = final_df.append(df_2, ignore_index=True)
    return final_df


def create_days_since_incorp_feature(features):
    """Processes the historical features from Tecton to create feature - 'days_since_incorp'
      Args:
        features (pd.DataFrame):  pandas dataframe with historical features of companies at various timestamps
      Returns:
        pd.DataFrame: Returns feature dataset with 'days_since_incorp' added
    """
    features['company_core_features_v3__incorporation_at'] = pd.to_datetime(
        features['company_core_features_v3__incorporation_at'])
    features['first_sar_created_dt'] = pd.to_datetime(features['first_sar_created_dt'])
    features.loc[features.first_sar_created_dt.isna(), 'days_since_incorp'] = features.loc[
                                                                                  features.first_sar_created_dt.isna(), 'window_end'] - \
                                                                              features.loc[
                                                                                  features.first_sar_created_dt.isna(), 'company_core_features_v3__incorporation_at']
    features.loc[~features.first_sar_created_dt.isna(), 'days_since_incorp'] = features.loc[
                                                                                   ~features.first_sar_created_dt.isna(), 'first_sar_created_dt'] - \
                                                                               features.loc[
                                                                                   ~features.first_sar_created_dt.isna(), 'company_core_features_v3__incorporation_at']
    features['days_since_incorp'] = features['days_since_incorp'].dt.days
    return features


def create_target_variables(features):
    """Processes the historical features from Tecton to create target labels - 'fincrime90_duration' & 'fincrime90_event'
     Args:
       features (pd.DataFrame):  pandas dataframe with historical features of companies at various timestamps
     Returns:
       pd.DataFrame: Returns feature dataset with target labels added
     """


    features["fincrime90_duration"] = features['first_sar_created_dt'] - features['timestamp']
    features["fincrime90_duration"] = features["fincrime90_duration"].dt.days
    features["fincrime90_duration"].fillna(90, inplace=True)
    features.loc[features["first_sar_created_dt"].isna(), 'fincrime90_event'] = 0
    features.loc[~features["first_sar_created_dt"].isna(), 'fincrime90_event'] = 1
    return features


# COMMAND ----------

def file_exists(dbfs_path_prefix, dataset_name):
    try:
        dbutils.fs.ls(os.path.join(dbfs_path_prefix, dataset_name))
        return True
    except Exception as e:
        if 'java.io.FileNotFoundException' in str(e):
            return False
        else:
            raise


def setup_tecton_dataset(dataset_name=None, workspace_name=None, feature_service_name=None):
    """Main method to run pipeline - create dataset spine, get historical features & save dataset
          Args:
            dataset_name (str):  name of the dataset using which we want to save the data
            workspace_name(str): Tecton workspace to be used
            feature_service_name(str): Feature service to be use dto obtain features from Tecton
      """
    if file_exists(dbfs_path_prefix, dataset_name + ".csv"):  # if dataset already present, exit without raising error
        dbutils.notebook.exit(os.path.join(dbfs_path_prefix, dataset_name + ".csv"))
    elif dataset_name in tecton.get_workspace('prod').list_datasets():
        dbutils.notebook.exit(dataset_name)
    else:
        # generate dataset spine
        print("LOADING SPINE")
        dataset_spine = setup_training_dataset_spine(start_year=startdate_year, start_month=startdate_month,
                                                     end_year=stopdate_year, end_month=stopdate_month,
                                                     multiple_day=multiple_days)
        # load features from tecton
        print("LOADING FEATURES FROM TECTON")
        features = get_data_from_fs(dataset_spine, "timestamp", feature_service_name, workspace_name)
        print(features)
        # load SAR labels from SQL
        print("LOADING SAR VALUES FROM SQL")
        member_sar_features = get_sar_labels(dataset_spine, features)
        print(member_sar_features)
        # process the SAR labels to obtain fincrime90_duration target variable
        print("PROCESSING SAR LABELS")
        member_sar_features = process_sar_dataframe(member_sar_features)
        print(member_sar_features)
        # combine features & SAR labels
        final_features = features.merge(member_sar_features, how="left", left_on=["company_id", "timestamp"],
                                        right_on=["company_id", "timestamp"])
        # create days_since_incorp feature
        final_features = create_days_since_incorp_feature(final_features)
        # create fincrime90_duration  & fincrime90_event column
        final_features = create_target_variables(final_features)
        # save dataset
        save_dataset(final_features, dataset_name)


# COMMAND ----------



# COMMAND ----------

workspace_name = "prod"
feature_service_name = dbutils.widgets.getArgument(name='feature_service_name')
dataset_name = setup_dataset_name(startdate_year, startdate_month, stopdate_year, stopdate_month)
setup_tecton_dataset(dataset_name, workspace_name, feature_service_name)
dbutils.notebook.exit(dataset_name)

# COMMAND ----------

feature_service_name

# COMMAND ----------

dbutils.notebook.exit(dataset_name)

# COMMAND ----------

