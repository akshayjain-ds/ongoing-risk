# Databricks notebook source
# MAGIC %run ./utils

# COMMAND ----------

# MAGIC %run ./feature_transformation

# COMMAND ----------

train_df = pd.concat([
  load_dataset('/dbfs/tmp/tm_app_victim_20230101_20230201').toPandas(),
  load_dataset('/dbfs/tmp/tm_app_victim_20230201_20230301').toPandas(),
  load_dataset('/dbfs/tmp/tm_app_victim_20230301_20230401').toPandas(),
  load_dataset('/dbfs/tmp/tm_app_victim_20230401_20230501').toPandas(),
  load_dataset('/dbfs/tmp/tm_app_victim_20230501_20230601').toPandas(),
  load_dataset('/dbfs/tmp/tm_app_victim_20230601_20230701').toPandas(),
],axis=0)

test_df = pd.concat([
  load_dataset('/dbfs/tmp/tm_app_victim_20230701_20230801').toPandas(),
  load_dataset('/dbfs/tmp/tm_app_victim_20230801_20230901').toPandas(),
  load_dataset('/dbfs/tmp/tm_app_victim_20230901_20231001').toPandas(),
],axis=0)

# dataset = load_dataset('/dbfs/tmp/tm_app_victim_20220501_20231001').toPandas()

# COMMAND ----------

train_df.shape, test_df.shape

# COMMAND ----------

TARGET_COLUMN = 'is_app_victim_counter_party'

# COMMAND ----------

def do_resampling(train_df: pd.DataFrame) -> pd.DataFrame:
  """
  Resampling of the training dataset
  """
  train_df = pd.concat(
    objs=[
      train_df[train_df[TARGET_COLUMN] == 0].sample(n=int(5e6)),
      train_df[train_df[TARGET_COLUMN] == 1]
    ],
    axis=0
  ).reset_index(drop=True)

  print(
    "Class imbalance in training after resampling: \n",
    train_df[TARGET_COLUMN].value_counts()
  )

  return train_df

# COMMAND ----------

train_df = do_resampling(train_df)
# test_df = do_resampling(test_df)

# COMMAND ----------

train_df.shape, test_df.shape

# COMMAND ----------

def transform_features(
  train_df: pd.DataFrame, test_df: pd.DataFrame
) -> (pd.DataFrame, pd.DataFrame):
  """
  Runs feature transformation.
  """
  print("Transforming features...")
  feature_transformer = FeatureTransformer()

  train_transformed_df = feature_transformer.fit_transform(train_df) # Fit and transform on train
  test_transformed_df = feature_transformer.transform(test_df) # Only transform test
  
  return train_transformed_df, test_transformed_df

# COMMAND ----------

train_df[['rule_feature_is_source_dest_clearlisted', 'rule_feature_source_dest_expiry_ts', 'rule_feature_source_dest_amount']] = -999
test_df[['rule_feature_is_source_dest_clearlisted', 'rule_feature_source_dest_expiry_ts', 'rule_feature_source_dest_amount']] = -999

# COMMAND ----------

train_transformed_df, test_transformed_df = transform_features(train_df, test_df)

# COMMAND ----------

train_transformed_df.shape, test_transformed_df.shape

# COMMAND ----------

train_transformed_df = train_transformed_df.astype(float)
test_transformed_df = test_transformed_df.astype(float)

# COMMAND ----------

# train_transformed_df[TARGET_COLUMN] = train_df[TARGET_COLUMN].values
# test_transformed_df[TARGET_COLUMN] = test_df[TARGET_COLUMN].values

# train_transformed_df['company_core_features_v3__is_registered'] = train_df['company_core_features_v3__is_registered'].values
# test_transformed_df['company_core_features_v3__is_registered'] = test_df['company_core_features_v3__is_registered'].values

# train_transformed_df['ifre_member_features_v2__industry_classification'] = train_df['ifre_member_features_v2__industry_classification'].values
# test_transformed_df['ifre_member_features_v2__industry_classification'] = test_df['ifre_member_features_v2__industry_classification'].values

# train_transformed_df['payment_recipient_events_features_new_to_tide__number_of_tide_accounts_with_same_payee'] = train_df['payment_recipient_events_features_new_to_tide__number_of_tide_accounts_with_same_payee'].values
# test_transformed_df['payment_recipient_events_features_new_to_tide__number_of_tide_accounts_with_same_payee'] = test_df['payment_recipient_events_features_new_to_tide__number_of_tide_accounts_with_same_payee'].values

# train_transformed_df['company_id'] = train_df['ifre_member_features_v2__industry_classification'].values
# test_transformed_df['company_id'] = test_df['company_id'].values

# COMMAND ----------

# save_dataset(spark.createDataFrame(train_transformed_df), "/dbfs/Users/sri.duddu@tide.co/train_transformed_df_llm", overwrite=True)

# COMMAND ----------

# save_dataset(spark.createDataFrame(test_transformed_df), "/dbfs/Users/sri.duddu@tide.co/test_transformed_df_llm", overwrite=True)

# COMMAND ----------

# MAGIC %md
# MAGIC <h3> GAN based Data Augmentation

# COMMAND ----------

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def apply_gan(X, y, num_steps=10000):
    """
    Apply GAN to generate synthetic samples for minority class.
    
    Parameters:
    - X: Feature matrix.
    - y: Target vector.
    
    Returns:
    - Resampled feature matrix and target vector.
    """
    
    # 1. Extract the minority class samples
    X_minority = X[y == 1]
    X_minority_tensor = torch.FloatTensor(X_minority)
    
    # 2. Define the GAN model layers
    generator_layers = nn.Sequential(
        nn.Linear(in_features=100, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=X_minority.shape[1]),
        nn.Sigmoid()
    )

    discriminator_layers = nn.Sequential(
        nn.Linear(in_features=X_minority.shape[1], out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=1),
        nn.Sigmoid()  # sigmoid => real or fake
    )

    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator_layers.parameters(), lr=0.001)
    optimizer_d = optim.Adam(discriminator_layers.parameters(), lr=0.0001)
    
    # 3. Train GAN
    for step in range(num_steps):
        # Train Discriminator
        optimizer_d.zero_grad()
        real_labels = torch.ones(len(X_minority_tensor), 1)
        fake_data = generator_layers(torch.randn(len(X_minority_tensor), 100))
        fake_labels = torch.zeros(len(X_minority_tensor), 1)
        
        logits_real = discriminator_layers(X_minority_tensor)
        logits_fake = discriminator_layers(fake_data.detach())
        
        loss_real = criterion(logits_real, real_labels)
        loss_fake = criterion(logits_fake, fake_labels)
        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()
        logits_fake = discriminator_layers(fake_data)
        loss_g = criterion(logits_fake, real_labels)
        loss_g.backward()
        optimizer_g.step()

    # 4. Generate synthetic samples
    num_synthetic_samples = int(len(X_minority)) * 5
    with torch.no_grad():
        synthetic_samples = generator_layers(torch.randn(num_synthetic_samples, 100)).numpy()

    # Append the synthetic samples to X and adjust y accordingly
    X_resampled = np.vstack([X, synthetic_samples])
    y_resampled = np.hstack([y, [1]*num_synthetic_samples])

    return X_resampled, y_resampled

# COMMAND ----------

# from tabgan.sampler import OriginalGenerator, GANGenerator, ForestDiffusionGenerator

# generate data
# new_train1, new_target1 = OriginalGenerator().generate_data_pipe(train_transformed_df, train_df[[TARGET_COLUMN]], test_transformed_df, )
# new_train2, new_target2 = GANGenerator().generate_data_pipe(train_transformed_df, train_df[[TARGET_COLUMN]], test_transformed_df, )
# new_train3, new_target3 = ForestDiffusionGenerator().generate_data_pipe(train_transformed_df, train_df[[TARGET_COLUMN]], test_transformed_df, )

# COMMAND ----------

X = train_transformed_df.values
y = train_df[TARGET_COLUMN]

# COMMAND ----------

X_resampled, y_resampled = apply_gan(X,y,10000)

# COMMAND ----------

np.unique(y_resampled, return_counts=True)

# COMMAND ----------

train_df_gan = pd.DataFrame(X_resampled, columns=list(train_transformed_df.columns))

# COMMAND ----------

train_df_gan.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Existing Model

# COMMAND ----------

best_params = {
  'iterations': 500,
  'depth': 10,
  'learning_rate': 0.01, 
  'l2_leaf_reg': 3, # l2 regularisation,  to 3 by default
  'auto_class_weights': 'SqrtBalanced', #Balanced
  'random_seed': 42, 
  'loss_function': 'Logloss',
  'eval_metric': 'PRAUC',
  'od_type': 'IncToDec',
  'od_pval': 0.01,
  'early_stopping_rounds': 40,
}

# COMMAND ----------

from catboost import CatBoostClassifier

model = CatBoostClassifier(verbose=False)
model.set_params(**best_params)

# COMMAND ----------

model.fit(train_transformed_df, y, sample_weight=train_transformed_df.requested_payment_value.astype(float))

# COMMAND ----------

#Select Threshold Custom Function
def select_threshold(proba, target, fpr_max = 0.1 ):
    # calculate roc curves
    fpr, tpr, thresholds = roc_curve(target, proba)
    # get the best threshold with fpr <=0.1
    best_treshold = thresholds[fpr <= fpr_max][-1]
    
    return best_treshold

# COMMAND ----------

y_prob = model.predict_proba(train_transformed_df)[:,1]

# COMMAND ----------

from sklearn.metrics import precision_recall_curve, roc_curve, precision_score, recall_score

# COMMAND ----------

target_fpr = 0.001
decision_threshold = select_threshold(y_prob, y, target_fpr)

# COMMAND ----------

print(decision_threshold)

# COMMAND ----------

test_df['prediction'] = (model.predict_proba(test_transformed_df)[:,1] > decision_threshold).astype(int)

# COMMAND ----------

test_df['prediction'].value_counts()

# COMMAND ----------

#KPI Metrics
def get_metrics(X, y, y_pred) -> dict:
    """Calculates metrics, both model and business,and returns them in a dictionary."""
    # print("Calculating metrics...")
    X_validate = X.copy()
    X_validate['is_anomaly'] = y
    X_validate['prediction'] = y_pred

    total_blocked_transactions = X_validate.loc[(X_validate.prediction == 1), 'prediction'].sum()
    
    #Transaction value based metrics
    total_funds_alerted_on = X_validate.loc[(X_validate.prediction == 1), 'requested_payment_value'].sum()
    total_fraud_funds = X_validate.loc[X_validate.is_anomaly == 1, 'requested_payment_value'].sum()
    total_fraud_funds_alerted_on = X_validate.loc[(X_validate.prediction == 1) & (X_validate.is_anomaly == 1), 'requested_payment_value'].sum()
    total_fraud_funds_missed = X_validate.loc[(X_validate.prediction == 0) & (X_validate.is_anomaly == 1), 'requested_payment_value'].sum()
    total_funds = X_validate['requested_payment_value'].sum()
    total_funds_approved = X_validate.loc[(X_validate.prediction == 0),'requested_payment_value'].sum()
    total_funds_false_positive = X_validate.loc[(X_validate.prediction == 1) & (X_validate.is_anomaly == 0), 'requested_payment_value'].sum()
    total_funds_true_negative = X_validate.loc[(X_validate.prediction == 0) & (X_validate.is_anomaly == 0), 'requested_payment_value'].sum()
    
    #Member based metrics
    total_members_true_positive = X_validate.loc[(X_validate.prediction == 1) & (X_validate.is_anomaly == 1), COMPANY_COL].nunique()
    total_members_false_positive = X_validate.loc[(X_validate.prediction == 1) & (X_validate.is_anomaly == 0), COMPANY_COL].nunique()
    total_members_true_negative = X_validate.loc[(X_validate.prediction == 0) & (X_validate.is_anomaly == 0), COMPANY_COL].nunique()
    total_members_false_negative = X_validate.loc[(X_validate.prediction == 0) & (X_validate.is_anomaly == 1), COMPANY_COL].nunique()
    total_members_not_alerted = X_validate.loc[(X_validate.is_anomaly == 0), COMPANY_COL].nunique()
    total_members_alerted = X_validate.loc[(X_validate.prediction == 1), COMPANY_COL].nunique()
    
    return {
      "# Blocked Txns": total_blocked_transactions,
      "Block Rate": (total_blocked_transactions * 100 / X_validate.shape[0]),
      "precision": precision_score(y, y_pred) * 100,
      "recall": recall_score(y, y_pred) * 100,
      "precision_by_value": (total_fraud_funds_alerted_on * 100 / total_funds_alerted_on),
      "recall_by_value": (total_fraud_funds_alerted_on * 100 / total_fraud_funds),
      "decline_rate": (total_funds_alerted_on * 100 / total_funds), 
      "fraud_exposure": (total_fraud_funds_missed * 100 / total_funds_approved),
      "false_positive_rate": (total_funds_false_positive * 100 / (total_funds_false_positive + total_funds_true_negative)),
      "review_rate": (total_members_false_positive * 100 / total_members_not_alerted),
      "percentage_of_funds_lost": (total_fraud_funds_missed * 100 / total_fraud_funds),
      "member_level_TPR": (total_members_true_positive * 100.0 / total_members_alerted),
      "member_level_Recall": (total_members_true_positive * 100.0 / (total_members_true_positive + total_members_false_negative))
    }

# COMMAND ----------

test_transformed_df['company_id'] = test_df['company_id'].values
get_metrics(test_transformed_df, test_df[TARGET_COLUMN].values, test_df['prediction'].values)

# COMMAND ----------

# MAGIC %md
# MAGIC GAN assisted Model

# COMMAND ----------

best_params = {
  'iterations': 500,
  'depth': 10,
  'learning_rate': 0.01, 
  'l2_leaf_reg': 3, # l2 regularisation,  to 3 by default
  'auto_class_weights': 'SqrtBalanced', #Balanced
  'random_seed': 42, 
  'loss_function': 'Logloss',
  'eval_metric': 'PRAUC',
  'od_type': 'IncToDec',
  'od_pval': 0.01,
  'early_stopping_rounds': 40,
}

# COMMAND ----------

from catboost import CatBoostClassifier

model = CatBoostClassifier(verbose=False)
model.set_params(**best_params)

# COMMAND ----------

model.fit(train_df_gan, y_resampled, sample_weight=train_df_gan.requested_payment_value.astype(float))

# COMMAND ----------

#Select Threshold Custom Function
def select_threshold(proba, target, fpr_max = 0.1 ):
    # calculate roc curves
    fpr, tpr, thresholds = roc_curve(target, proba)
    # get the best threshold with fpr <=0.1
    best_treshold = thresholds[fpr <= fpr_max][-1]
    
    return best_treshold

# COMMAND ----------

y_prob = model.predict_proba(train_df_gan)[:,1]

# COMMAND ----------

from sklearn.metrics import precision_recall_curve, roc_curve, precision_score, recall_score

# COMMAND ----------

target_fpr = 0.001
decision_threshold = select_threshold(y_prob, y_resampled, target_fpr)

# COMMAND ----------

print(decision_threshold)

# COMMAND ----------

test_df['prediction'] = (model.predict_proba(test_transformed_df)[:,1] > decision_threshold).astype(int)

# COMMAND ----------

test_df['prediction'].value_counts()

# COMMAND ----------

#KPI Metrics
def get_metrics(X, y, y_pred) -> dict:
    """Calculates metrics, both model and business,and returns them in a dictionary."""
    # print("Calculating metrics...")
    X_validate = X.copy()
    X_validate['is_anomaly'] = y
    X_validate['prediction'] = y_pred

    total_blocked_transactions = X_validate.loc[(X_validate.prediction == 1), 'prediction'].sum()
    
    #Transaction value based metrics
    total_funds_alerted_on = X_validate.loc[(X_validate.prediction == 1), 'requested_payment_value'].sum()
    total_fraud_funds = X_validate.loc[X_validate.is_anomaly == 1, 'requested_payment_value'].sum()
    total_fraud_funds_alerted_on = X_validate.loc[(X_validate.prediction == 1) & (X_validate.is_anomaly == 1), 'requested_payment_value'].sum()
    total_fraud_funds_missed = X_validate.loc[(X_validate.prediction == 0) & (X_validate.is_anomaly == 1), 'requested_payment_value'].sum()
    total_funds = X_validate['requested_payment_value'].sum()
    total_funds_approved = X_validate.loc[(X_validate.prediction == 0),'requested_payment_value'].sum()
    total_funds_false_positive = X_validate.loc[(X_validate.prediction == 1) & (X_validate.is_anomaly == 0), 'requested_payment_value'].sum()
    total_funds_true_negative = X_validate.loc[(X_validate.prediction == 0) & (X_validate.is_anomaly == 0), 'requested_payment_value'].sum()
    
    #Member based metrics
    total_members_true_positive = X_validate.loc[(X_validate.prediction == 1) & (X_validate.is_anomaly == 1), COMPANY_COL].nunique()
    total_members_false_positive = X_validate.loc[(X_validate.prediction == 1) & (X_validate.is_anomaly == 0), COMPANY_COL].nunique()
    total_members_true_negative = X_validate.loc[(X_validate.prediction == 0) & (X_validate.is_anomaly == 0), COMPANY_COL].nunique()
    total_members_false_negative = X_validate.loc[(X_validate.prediction == 0) & (X_validate.is_anomaly == 1), COMPANY_COL].nunique()
    total_members_not_alerted = X_validate.loc[(X_validate.is_anomaly == 0), COMPANY_COL].nunique()
    total_members_alerted = X_validate.loc[(X_validate.prediction == 1), COMPANY_COL].nunique()
    
    return {
      "# Blocked Txns": total_blocked_transactions,
      "Block Rate": (total_blocked_transactions * 100 / X_validate.shape[0]),
      "precision": precision_score(y, y_pred) * 100,
      "recall": recall_score(y, y_pred) * 100,
      "precision_by_value": (total_fraud_funds_alerted_on * 100 / total_funds_alerted_on),
      "recall_by_value": (total_fraud_funds_alerted_on * 100 / total_fraud_funds),
      "decline_rate": (total_funds_alerted_on * 100 / total_funds), 
      "fraud_exposure": (total_fraud_funds_missed * 100 / total_funds_approved),
      "false_positive_rate": (total_funds_false_positive * 100 / (total_funds_false_positive + total_funds_true_negative)),
      "review_rate": (total_members_false_positive * 100 / total_members_not_alerted),
      "percentage_of_funds_lost": (total_fraud_funds_missed * 100 / total_fraud_funds),
      "member_level_TPR": (total_members_true_positive * 100.0 / total_members_alerted),
      "member_level_Recall": (total_members_true_positive * 100.0 / (total_members_true_positive + total_members_false_negative))
    }

# COMMAND ----------

test_transformed_df['company_id'] = test_df['company_id'].values
get_metrics(test_transformed_df, test_df[TARGET_COLUMN].values, test_df['prediction'].values)