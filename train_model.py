# Databricks notebook source
# MAGIC %md 
# MAGIC
# MAGIC <h1>Train Ongoing Risk Model</h1>
# MAGIC
# MAGIC This notebook is responsible for training the ongoing risk model and logging the mlflow run and registering the model.
# MAGIC
# MAGIC Step1: Preprocess and validate the training dataset <br>
# MAGIC
# MAGIC Step2: Perform MLFlow run, train model and log the artifacts and metrics <br>
# MAGIC
# MAGIC Step3: Register the model on MLFlow and return the MLFlowVersion object
# MAGIC

# COMMAND ----------

# MAGIC %md ***Setup dependencies***

# COMMAND ----------

# MAGIC %md ***Get Widget Values***

# COMMAND ----------

# MAGIC %md ***Import Model Definitions and Utils***

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

# dbutils.widgets.removeAll()
# dbutils.widgets.text("dbfs_path_prefix", "dbfs:/tmp")
# dbutils.widgets.text("training_dataset", "ofre_dummy.csv")
# dbutils.widgets.text("experiment_name", "/Users/aravind.maguluri@tide.co/mlflow_experiments/ongoing-risk-temp")
# dbutils.widgets.text("model_name", "ongoing_risk_debug_am")
# dbutils.widgets.dropdown("log_shap", "No", ["Yes", "No"])
dbfs_path_prefix = dbutils.widgets.getArgument(name='dbfs_path_prefix')
training_dataset_name = dbutils.widgets.getArgument(name='training_dataset')
experiment_name = dbutils.widgets.getArgument(name='experiment_name')
model_name = dbutils.widgets.getArgument(name='model_name')
log_shap = dbutils.widgets.getArgument(name='log_shap')
print(dbfs_path_prefix, training_dataset_name)

# COMMAND ----------

# MAGIC %run ./feature_transformation

# COMMAND ----------

# MAGIC %run ./model_definition

# COMMAND ----------

# MAGIC %md ***Import libraries***

# COMMAND ----------

import warnings, shap, tempfile
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

# from pandas_profiling import ProfileReport
warnings.filterwarnings("ignore")


def validate_dataset(input_df: pd.DataFrame) -> None:
    """Method to check if the dataset received from tecton has valid data and doesn't have issues with materialization

    Args:
        input_df (pd.DataFrame): Input dataframe for validation

    Returns: None
    """
    # Validate if mandatory columns are present
    missing_cols = set(FEATURES + (EVENT_COL, DURATION_COL,)).difference(set(input_df.columns))
    if len(missing_cols) > 0:
        raise Exception(f"Input data missing required columns - {missing_cols}")

    # Validate if any columns have unexpected null values
    null_cols = set(input_df.columns[input_df.isnull().sum() > 0]).difference(
        set(DROP_NA_COLS).union(set(IMPUTER_COLS))).intersection(set(FEATURES)).difference(IFRE_COLS)
    if len(null_cols) > 0:
        raise Exception(f"Null values found in not-null columns - {null_cols}")

    # Validate if numerical data present in numerical feature columns
    if input_df[list(NUMERICAL_COLS)].apply(lambda x: x.dtype not in ['float64', 'int64']).sum() > 0:
        raise Exception(f"Invalid data found in the numerical columns")

    # Validate if label columns are numeric
    if input_df[list(EVENT_COLS)].apply(lambda x: x.dtype not in ['float64', 'int64']).sum() > 0:
        raise Exception(f"Non-numerical data found in label columns")

    # Validate if event column has valid values
    if len(set(input_df[EVENT_COL]).difference(set([0, 1]))) > 0:
        raise Exception(f"Event column has invalid values")

    # Validate if duration column has all integer values
    if input_df[DURATION_COL].dtype == 'float64' and (not all(x.is_integer() for x in input_df[DURATION_COL])):
        raise Exception(f"Duration column has non-integer values")

    # Validate if duration column has valid values
    if input_df[(input_df[DURATION_COL] > 90) | (input_df[DURATION_COL] < 1)].shape[0]:
        raise Exception(f"Duration column has invalid values")


# Validate if numerical data present in IFRE feature columns
# if input_df[list(IFRE_COLS)].apply(lambda x: x.dtype in ['float64', 'int64']).sum() > 0:
#	raise Exception(f"Invalid data found in the IFRE columns")

def save_eda_report(training_dataset: pd.DataFrame) -> None:
    """Method to save EDA on a dataset using pandas profiling report

    Args:
        training_dataset (pd.DataFrame): Input data for creating EDA report

    Returns: None
    """
    with tempfile.NamedTemporaryFile(prefix="EDA_", suffix=".html") as data_overview_file:
        data_overview_file_name = data_overview_file.name
        profile = ProfileReport(training_dataset, minimal=True)
        profile.to_file(output_file=data_overview_file_name)
        mlflow.log_artifact(data_overview_file_name, "Explorary Data Analysis")


def save_probability_calibration(model: object, training_dataset: pd.DataFrame) -> None:
    """Method to save probability calibration to a file

    Args:
        model (object): Fitted model for which the summary is saved
        training_dataset (pd.DataFrame): Input data

    Returns: None
    """
    with tempfile.NamedTemporaryFile(prefix="probability_calibration_", suffix=".png") as prob_file:
        from scipy.stats import boxcox

        prob_file_name = prob_file.name
        X = preprocess_dataset(training_dataset)
        X_preprocessed = model.feature_selection.transform(model.feature_transformer.transform(X.loc[:, FEATURES]))
        rsf_prob = model.uncalibrated_model.predict_survival_function(
            X_preprocessed.loc[:, X_preprocessed.columns != DURATION_COL], return_array=True)[:, -1]
        calibrated_rsf_prob = model.calibrated_model.predict_proba(
            X_preprocessed.loc[:, X_preprocessed.columns != DURATION_COL])[:, 0]
        y_test = X.loc[:, list(EVENT_COLS)]
        y_test.fincrime90_duration = (y_test.fincrime90_duration / 7).round()
        y_test = y_test.loc[:, [EVENT_COL]]

        fig, axes = plt.subplots(1, 2, figsize=(20, 5))
        ax = axes[0]
        sns.distplot(1 - calibrated_rsf_prob, label="Calibrated", ax=ax)
        sns.distplot(1 - rsf_prob, label="Uncalibrated", ax=ax)
        ax.legend()
        ax.set_title("Hazard Probability - Random Survival Forest")

        ax = axes[1]
        ax.plot([0, 1], [0, 1], linestyle='--', color='black')  # plot perfectly calibrated
        fop_true, prob_pred_rsf = calibration_curve(y_test[EVENT_COL], 1 - rsf_prob, n_bins=10, normalize=True)
        ax.plot(prob_pred_rsf, fop_true, marker='.', label="Uncalibrated")
        fop_true, prob_pred_rsf_cal = calibration_curve(y_test[EVENT_COL], 1 - calibrated_rsf_prob, n_bins=10,
                                                        normalize=True)
        ax.plot(prob_pred_rsf_cal, fop_true, marker='.', label="Calibrated")
        ax.legend()
        ax.set_title("Calibration curve - Random Survival Forest")

        plt.tight_layout()
        plt.show()
        fig.savefig(prob_file_name)
        mlflow.log_artifact(prob_file_name, "Probability Calibration")


def save_feature_effect_summary(model: object, shap_values: np.array, X: pd.DataFrame, filtered_cols) -> None:
    """Method to save summarize the effects of all the features

      Args:
          model (object): Fitted model
          shap_values (np.array): shap_values
          X (pd.DataFrame): Input Dataframe

      Returns: None
    """
    with tempfile.NamedTemporaryFile(prefix="Feature_Effect_Summary", suffix=".png") as feature_effect_summary:
        feature_effect_summary_file_name = feature_effect_summary.name
        fig = plt.figure(figsize=(20, 10))
        shap.summary_plot(shap_values[0], X, feature_names=filtered_cols, max_display=30, show=False, plot_type="dot")
        plt.title('Feature Effect(SHAP) Summary')
        plt.savefig(feature_effect_summary_file_name, bbox_inches='tight')
        plt.close(fig)
        mlflow.log_artifact(feature_effect_summary_file_name,
                            "Summary of the effects(SHAP) of features")  # Upload Summary of the effects


def save_partial_dependence_plot(model: object, shap_values: np.array, X: pd.DataFrame, filtered_cols) -> None:
    """Method to save partial dependence plot

      Args:
          model (object): Fitted model
          shap_values (np.array): shap_values
          X (pd.DataFrame): Input Dataframe

      Returns: None
    """
    with tempfile.NamedTemporaryFile(prefix="Partial_Dependence_Plot", suffix=".png") as partial_dependence_plot:
        partial_dependence_plot_file_name = partial_dependence_plot.name
        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(40, 30))
        features = X.columns
        mask = np.abs(shap_values[0]).mean(0).argsort()[::-1]
        feature_list = list(features[mask])[:12]
        row_cnt = col_cnt = 0
        for each_feature in feature_list:
            shap.dependence_plot(each_feature, shap_values[0], X, ax=axes[row_cnt][col_cnt], show=False)
            col_cnt = col_cnt + 1
            if col_cnt == 4:
                col_cnt = 0
                row_cnt = row_cnt + 1
        plt.title('Partial Dependence Plot(SHAP)')
        plt.savefig(partial_dependence_plot_file_name, bbox_inches='tight')
        plt.close(fig)
        mlflow.log_artifact(partial_dependence_plot_file_name,
                            "Partial Dependence Plot")  # Upload Partial Dependence Plot


def save_force_plot(model: object, explainer: shap.KernelExplainer, shap_values: np.array, X: pd.DataFrame) -> None:
    """Method to save force plot(SHAP)

      Args:
          model (object): Fitted model
          shap_values (np.array): shap_values
          X (pd.DataFrame): Input Dataframe

      Returns: None
    """
    with tempfile.NamedTemporaryFile(prefix="Force_Plot1", suffix=".png") as force_plot:
        force_plot_file_name = force_plot.name
        shap.force_plot(explainer.expected_value[0], shap_values[0][0, :], X.iloc[0, :], matplotlib=True, show=False)
        plt.title('Force Plot(SHAP) 1')
        plt.savefig(force_plot_file_name, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(force_plot_file_name, "Force Plot 1")  # Upload Force Plot

    with tempfile.NamedTemporaryFile(prefix="Force_Plot2", suffix=".png") as force_plot:
        force_plot_file_name = force_plot.name
        shap.force_plot(explainer.expected_value[0], shap_values[0][1, :], X.iloc[1, :], matplotlib=True, show=False)
        plt.title('Force Plot(SHAP) 2')
        plt.savefig(force_plot_file_name, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(force_plot_file_name, "Force Plot 2")  # Upload Force Plot


def shap_feature_importance_bar_plot(model: object, shap_values: np.array, X: pd.DataFrame, filtered_cols) -> None:
    with tempfile.NamedTemporaryFile(prefix="Shap_Feature_Importance_Bar_Plot",
                                     suffix=".png") as shap_feature_importance_bar_plot:
        shap_feature_importance_bar_plot_file_name = shap_feature_importance_bar_plot.name
        fig = plt.figure(figsize=(20, 10))
        shap.summary_plot(shap_values, X, feature_names=filtered_cols, max_display=30, show=False, plot_type='bar')
        plt.title('SHAP Feature Importance - Bar Plot')
        plt.savefig(shap_feature_importance_bar_plot_file_name, bbox_inches='tight')
        plt.close(fig)
        mlflow.log_artifact(shap_feature_importance_bar_plot_file_name,
                            "Summary of the effects(SHAP) of features")  # Upload Summary of the effects as bar plot


def calculate_risk_rating(input_df):
    hazard_prob = 1 - ongoing_risk.model.calibrated_model.predict_proba(input_df)
    map_hazard_risk = lambda x: 1 if x <= 0.005 else (3 if x >= 0.01 else 2)  # 1 - Low, 2 - Medium, 3 - High
    vfunc = np.vectorize(map_hazard_risk)
    return vfunc(hazard_prob).astype(np.int)


def get_shap_values(sample_data, ongoing_risk):
    sample_data_preprocess = preprocess_dataset(sample_data)
    sample_data = ongoing_risk.model.feature_selection.transform(
        ongoing_risk.model.feature_transformer.transform(sample_data_preprocess.loc[:, FEATURES]))
    filtered_cols = list(sample_data.columns)
    explainer = shap.KernelExplainer(model=calculate_risk_rating, data=shap.sample(sample_data, 100),
                                     feature_names=filtered_cols, link="identity", keep_index=True)
    shap_values = explainer.shap_values(sample_data.head(1000))
    return shap_values, sample_data, filtered_cols, explainer


# COMMAND ----------

# MAGIC %md Load Dataset

# COMMAND ----------

import os
import pandas as pd


def file_exists(dataset_name):
    try:
        dbutils.fs.ls(os.path.join(dbfs_path_prefix, dataset_name))
        return True
    except Exception as e:
        if 'java.io.FileNotFoundException' in str(e):
            return False
        else:
            raise


def load_dataset(dataset_name: str):
    return pd.read_csv(os.path.join(dbfs_path_prefix.replace("dbfs:", "/dbfs"), dataset_name), low_memory=False)


if file_exists(training_dataset_name):
    training_dataset = load_dataset(training_dataset_name)
else:
    raise Exception('Dataset unavailable !!!')

# COMMAND ----------

# MAGIC %md Preprocess and validate dataset

# COMMAND ----------

# Preprocess dataset to format column names, remove records with null values for mandatory columns
training_dataset = preprocess_dataset(training_dataset)

# Validate dataset before using for training, passes sliently if successful, throws exception in case of issue
validate_dataset(training_dataset)

# COMMAND ----------

# MAGIC %md Run Experiments

# COMMAND ----------

mlflow_run_id = None
tags = {
    "model_type": "RandomSurvivalForest",
    "model_library": "scikit-survival",
    "tecton_dataset": training_dataset_name
}

# load areas_list .csv file
areas_list = pd.read_csv('/dbfs/FileStore/FileStore/areas.csv', header=None, names=['area_code']).loc[:,
             'area_code'].values

# Set an experiment name, which must be unique and case sensitive.
mlflow.set_experiment(experiment_name)
# Log model to local directory
with mlflow.start_run(run_name=model_name) as run:
    mlflow.set_tags(tags)  # Set tags on the mlflow run
    ongoing_risk = MLFlowWrapper(
        OngoingRisk(areas_list, cv_splits=5, cv_repeats=1, test_size=0.2, random_seed=123))  # Train the model
    return_value = ongoing_risk.fit(training_dataset)
    print(return_value)
    mlflow.log_metrics(metrics=ongoing_risk.model.metrics)
    mlflow.log_params(params=ongoing_risk.model.best_params)
    mlflow.pyfunc.log_model(model_name, python_model=ongoing_risk)  # Log model on mlflow
    print(log_shap)
    if log_shap == "Yes":
        training_data_shap_sample = training_dataset.sample(1000, random_state=123)
        shap_values, sample_data, filtered_cols, explainer = get_shap_values(training_data_shap_sample, ongoing_risk)
        save_feature_effect_summary(ongoing_risk.model.calibrated_model, shap_values, sample_data.head(100),
                                    filtered_cols)  # Upload Feature Effect Summary (SHAP)
        feature_list = save_partial_dependence_plot(ongoing_risk.model.calibrated_model, shap_values,
                                                    sample_data.head(100),
                                                    filtered_cols)  # Upload Partial Depedence Plot (SHAP)
        shap_feature_importance_bar_plot(ongoing_risk.model.calibrated_model, shap_values, sample_data.head(100),
                                         filtered_cols)
        save_force_plot(ongoing_risk.model.calibrated_model, explainer, shap_values,
                        sample_data.head(100))  # Upload Force Plot (SHAP)
    # save_eda_report(training_dataset) # Upload EDA
    save_probability_calibration(ongoing_risk.model, training_dataset)  # Upload probability calibration
    mlflow_run_id = run.info.run_id  # Store the run-id to return

# COMMAND ----------

# MAGIC %md ***Register Model***

# COMMAND ----------

model_version = mlflow.register_model(f"runs:/{mlflow_run_id}/{model_name}", model_name)
dbutils.notebook.exit(model_version)

# COMMAND ----------

