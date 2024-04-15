# Databricks notebook source
# MAGIC %md 
# MAGIC
# MAGIC <h1>Ongoing Risk - Evaluate Model</h1>
# MAGIC
# MAGIC This notebook is responsible for evaluating model performance over a period of time with detailed visualizations.
# MAGIC
# MAGIC
# MAGIC Step1: Setup datasets for each month in the validation time period <br>
# MAGIC
# MAGIC Step2: Predict risk ratings and provide visualizations to evaluate the performance of the model for each month <br>
# MAGIC
# MAGIC Step3: Provide a trend of how the model is performing for different thresholds of risk groups <br>

# COMMAND ----------

# MAGIC %md ***Setup dependencies***

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

# MAGIC %md ***Load Widget values***

# COMMAND ----------

from dateutil.relativedelta import *
import tecton
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
% matplotlib
inline

dbutils.widgets.removeAll()
dbutils.widgets.text("model_name", "ongoing_risk_debug_am")
dbutils.widgets.text("version", "0")
dbutils.widgets.text("dbfs_path_prefix", "dbfs:/tmp")
dbutils.widgets.text("valid_data", "csv")

model_name = dbutils.widgets.getArgument(name='model_name')
model_version = dbutils.widgets.getArgument(name='version')
dbfs_path_prefix = dbutils.widgets.getArgument(name='dbfs_path_prefix')
valid_data = dbutils.widgets.getArgument(name='valid_data')

import datetime


# COMMAND ----------

# MAGIC %md ***Import libraries***

# COMMAND ----------

# MAGIC %run ./feature_transformation

# COMMAND ----------

# MAGIC %run ./model_definition

# COMMAND ----------

# MAGIC %md ***Setup datasets for each month of evaluation***

# COMMAND ----------

def load_dataset(dataset_name: str):
    return pd.read_csv(os.path.join(dbfs_path_prefix.replace("dbfs:", "/dbfs"), dataset_name), low_memory=False)


valid_dataset = load_dataset(valid_data)

# COMMAND ----------

valid_dataset['month_year'] = pd.to_datetime(valid_dataset['timestamp']).dt.year.astype(str) + '_' + pd.to_datetime(
    valid_dataset['timestamp']).dt.month.astype(str)

# COMMAND ----------

list_of_datasets = []
for i in valid_dataset['month_year'].unique():
    list_of_datasets.append(valid_dataset[valid_dataset['month_year'] == i].reset_index(drop=True))

# COMMAND ----------

# MAGIC %md ***Load Model from Registry***

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
mlflow_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

# COMMAND ----------

# MAGIC %md ***Get predictions and stats for the validation period***

# COMMAND ----------

risk_df_all = pd.DataFrame()
member_pct_trend_df = pd.DataFrame()
fincrime_trend_df = pd.DataFrame()

for month_df in list_of_datasets:
    print(month_df['timestamp'].value_counts())
    month_id = month_df.month_year.unique()[0]
    validation_df = preprocess_dataset(month_df)
    X_test, y_test = validation_df.loc[:, FEATURES + ('company_id',)], validation_df.loc[:, list(EVENT_COLS)]
    risk_df = mlflow_model.predict(X_test)
    predictions_df = pd.merge(validation_df, risk_df, on=["company_id"])[
        ['fincrime90_event', 'company_id', 'timestamp', 'survival_probability', 'risk_rating']]
    print(predictions_df)
    predictions_df.to_csv("/dbfs/tmp/ofre-retraining-2023/DD-4462/predictions/" + str(model_name) + '_' + str(
        model_version) + "_predictions_" + month_id + "_.csv")
    risk_df_all = risk_df_all.append(risk_df)
    # risk_level_pct, risk_level_fincrime = OngoingRisk.get_risk_stats(risk_df.survival_probability, y_test.loc[:, EVENT_COL], low_risk_threshold=0.0024, high_risk_threshold=0.043)
    risk_level_pct, risk_level_fincrime = OngoingRisk.get_risk_stats(risk_df.survival_probability,
                                                                     y_test.loc[:, EVENT_COL],
                                                                     low_risk_threshold=0.0031,
                                                                     high_risk_threshold=0.03)  # using the thresholds from daily prediction jobs v2 t6
    member_pct_trend_df = member_pct_trend_df.append(
        pd.Series([month_id, risk_level_pct['High'], risk_level_pct['Medium'], risk_level_pct['Low']],
                  index=['Month', 'High', 'Medium', 'Low']), ignore_index=True)
    fincrime_trend_df = fincrime_trend_df.append(
        pd.Series([month_id, risk_level_fincrime['High'], risk_level_fincrime['Medium'], risk_level_fincrime['Low']],
                  index=['Month', 'High', 'Medium', 'Low']), ignore_index=True)

# COMMAND ----------

fincrime_trend_df.set_index('Month', inplace=True)
avg_values = fincrime_trend_df.mean()
std_values = fincrime_trend_df.std()
fincrime_trend_df.loc['Average'] = avg_values
fincrime_trend_df.loc['Std'] = std_values
fincrime_trend_df.T.round(3)

# COMMAND ----------

member_pct_trend_df.set_index('Month', inplace=True)
avg_values = member_pct_trend_df.mean()
std_values = member_pct_trend_df.std()
member_pct_trend_df.loc['Average'] = avg_values
member_pct_trend_df.loc['Std'] = std_values
member_pct_trend_df.T.round(3)

# COMMAND ----------

# MAGIC %md ***Trend of FinCrime% over the validation period for risk groups***

# COMMAND ----------

plt.rcParams["figure.figsize"] = (10, 5)

fincrime_trend_df.loc[:, 'Low'].iloc[:-2].plot(marker='o', color='green', linestyle='--', label='Low-Risk')
fincrime_trend_df.loc[:, 'Medium'].iloc[:-2].plot(marker='o', color='blue', linestyle='--', label='Medium-Risk')
fincrime_trend_df.loc[:, 'High'].iloc[:-2].plot(marker='o', color='red', linestyle='--', label='High-Risk')
months = list(fincrime_trend_df.index[:-2].values)

plt.ylabel('FinCrime%')
plt.title('FinCrime% - Low / Medium / High Risk Groups')
plt.legend(loc='upper left')
plt.grid(which='major', linestyle='-', linewidth=0.3)
plt.yticks(np.arange(0, 20, 1), [round(x, 1) for x in np.arange(0, 20, 1)])
display()

# COMMAND ----------

# MAGIC %md ***Trend of Member% over the validation period for risk groups***

# COMMAND ----------

plt.rcParams["figure.figsize"] = (20, 5)
member_pct_trend_df.iloc[:-2, :].plot(kind='bar', stacked=True, rot=90)
plt.ylabel('Member%')
plt.title('Member% - Low / Medium / High Risk Groups')
plt.legend(loc='best')
plt.grid(which='major', linestyle='-', linewidth=0.3)
display()