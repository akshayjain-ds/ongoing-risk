# Databricks notebook source
# MAGIC %run ./set_up

# COMMAND ----------

# MAGIC %pip install imbalanced-learn
# MAGIC %pip install torch

# COMMAND ----------

# MAGIC %run ./train_metadata

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
from datetime import timedelta
import tecton
from optbinning import OptimalBinning
from sklearn.preprocessing import MinMaxScaler
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import gc
gc.enable()

# COMMAND ----------

day_of_month = 'fixed_day'
target_b, target_c, target_d = 'is_app_fraud_45d', 'app_fraud_amount_45d', 'days_to_fraud_45d'

features = pd.read_csv(f"/Workspace/Shared/Decisioning/Strawberries/ongoing-ntt_mp_risk-v1/artefacts/{'ntt_ftt' if include_ftt else 'ntt'}_{day_of_month}_encoded_features_{train_start_date}_{val_end_date}.csv.gz", memory_map=True, dtype={id1: str})

features = features.assign(
  timestamp = lambda col: pd.to_datetime(col[timestamp]),
)
features.sort_values(by=[id1, timestamp], inplace=True)
features.shape

# COMMAND ----------

(features['is_ntt']>0).mean(), (features['is_ftt']>0).mean()

# COMMAND ----------

features.groupby(['is_ntt', 'is_ftt'])[target_b].agg(['sum', 'mean'])

# COMMAND ----------

features[[target_d, target_c]].mean()

# COMMAND ----------

# markers = []
# for id, series in features.groupby([id1])[target_b]:
  
#   marker = []
#   counter = 0
  
#   for b in series.to_list():

#     if b == 0:

#       if counter == 0:
#         marker.append(True)
#       else:
#         marker.append(False)
    
#     else:

#       if counter in [0,1]:
#         marker.append(True)
#         counter +=1
#       else:
#         marker.append(False)    
  
#   markers += marker
  
# features = features[markers]
# features.shape

# COMMAND ----------

(features['is_ntt']>0).mean(), (features['is_ftt']>0).mean()

# COMMAND ----------

features.groupby(['is_ntt', 'is_ftt'])[target_b].agg(['sum', 'mean'])

# COMMAND ----------

features[[target_d, target_c]].mean()

# COMMAND ----------


id_is_app_fraud_45d = features.groupby([id1], as_index=False)[target_b].max()
companies = id_is_app_fraud_45d[id1]
is_app_fraud_45d = id_is_app_fraud_45d[target_b]*1

train_ids, test_ids = train_test_split(companies, stratify=is_app_fraud_45d, 
                                       test_size=0.25, random_state=seed)
train_ids.shape, test_ids.shape

# COMMAND ----------

feature_list = [
 'attribution_marketing_channel_encoded',
 'company_postcode_encoded',
 'company_structurelevelwise_1_encoded',
 'company_directors_count_encoded',
 'applicant_nationality_encoded',
 'applicant_postcode_encoded',
 'applicant_idcountry_issue_encoded',
 'applicant_id_type_encoded',
 'applicant_email_domain_encoded',
 'company_nob_encoded',
 'company_age_at_timestamp_encoded',
 'applicant_years_to_id_expiry_encoded',
 'fastpmt_beneficiaries',
 'max_cash_deposits',
 'high_pmt_out',
 'invoice_matched_before_timestamp',
 'fastpmt_benefactors',
 'high_card_pmts',
 'days_to_approval',
 'applicant_age_at_completion',
 'ddebit_wtd_pct',
 'cashtxns_latehrs',
 'company_is_registered',
 'high_fpmt_in',
 'pct_round_txns',
 'ddebit_beneficiaries',
 'card_pans_cnt',
 'applicant_email_numeric',
 'avg_withdrawal',
 'outpmt_wtd_pct',
 'cardpmt_wtd_pct',
 'is_ntt',
 'cardwtd_wtd_pct',
 'max_withdrawal',
 'days_remaining_as_ntt_ftt',
 'receipt_match_before_timestamp',
 'high_ddebit',
 'fastpmt_wtd_pct',
 'high_fpmt_out',
 'receipt_uploaded_before_timestamp',
 'deposit_wtd_frequency_ratio',
 'is_restricted_keyword_present',
 'pos_atm_locations',
 'invoice_chased_before_timestamp',
 'is_ftt',
 'high_card_wtds',
 'tester_pmt_cnt',
 'avg_deposit',
 'cardpmt_acceptors',
 'max_deposit',
 'first_invoice_before_timestamp',
 'xero_txns_cnt',
 'pct_unique_txns',
 'high_pmt_in',
 'days_on_books',
 'hmrc_txns_cnt',
 'inpmt_dep_pct'
 ]
len(feature_list)

# COMMAND ----------

X_train = features[features[id1].isin(train_ids.tolist())][feature_list]
X_test = features[features[id1].isin(test_ids.tolist())][feature_list]

y_train = features[features[id1].isin(train_ids.tolist())][target_b]
y_test = features[features[id1].isin(test_ids.tolist())][target_b]

ftt_train = features[features[id1].isin(train_ids.tolist())]['is_ftt'] > 0
ftt_test = features[features[id1].isin(test_ids.tolist())]['is_ftt'] > 0

X_min_train = X_train[y_train==1]
y_min_train = y_train[y_train==1]

X_maj_train = X_train[y_train==0]
y_maj_train = y_train[y_train==0]

X_train.shape, y_train.shape, ftt_train.shape, X_test.shape, y_test.shape, ftt_test.shape

# COMMAND ----------

X_min_train.shape, y_min_train.shape, X_maj_train.shape, y_maj_train.shape

# COMMAND ----------

print(np.average(y_train),
      np.average(y_test))

# COMMAND ----------

print(np.average(y_train[ftt_train]),
      np.average(y_test[ftt_test]))

# COMMAND ----------

print(np.average(y_train[~ftt_train]),
      np.average(y_test[~ftt_test]))

# COMMAND ----------

from imblearn.over_sampling import BorderlineSMOTE, KMeansSMOTE, SVMSMOTE

# COMMAND ----------

smote = BorderlineSMOTE(random_state=seed, sampling_strategy=0.1)
X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X_train, y_train)
X_train_SMOTE.shape, y_train_SMOTE.shape

# COMMAND ----------

import torch
torch.use_deterministic_algorithms(True)
import torch.nn as nn
import torch.optim as optim
import numpy as np
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
torch.manual_seed(seed)

def get_generator_block(input_dim, output_dim):
  
  return nn.Sequential(
      nn.Linear(input_dim, output_dim),
      nn.BatchNorm1d(output_dim),
      nn.LeakyReLU(0.2, inplace=True),
  )

def get_discriminator_block(input_dim, output_dim):
  
  return nn.Sequential(
      nn.Linear(input_dim, output_dim),
      nn.LeakyReLU(0.2, inplace=True)        
  )

def early_stop(loss: list, tol=1e-04, patince=3)-> bool:

  if len(loss) <= patince:
    return False
  
  else:
    return all([loss[-3] - loss[-2] < tol,
                loss[-2] - loss[-1] < tol])

def shuffle_in_unison(a, b): 
  
  assert len(a) == len(b)
  shuffled_a = np.empty(a.shape, dtype=a.dtype) 
  shuffled_b = np.empty(b.shape, dtype=b.dtype)
  permutation = np.random.permutation(len(a))
  
  for old_index, new_index in enumerate(permutation):
      shuffled_a[new_index] = a[old_index]
      shuffled_b[new_index] = b[old_index]
  
  return shuffled_a, shuffled_b

def apply_smote_gan(X_min_real, y_min_real, X_min_fake, y_min_fake, X_maj_real, y_maj_real, 
                    epochs=100):

    # 1. Extract the Minority class samples and Smote samples
    if isinstance(X_min_real, pd.DataFrame):
      X_min_real = X_min_real.values

    if isinstance(y_min_real, pd.DataFrame) or isinstance(y_min_real, pd.Series):
      y_min_real = y_min_real.values.reshape(-1,1)

    if isinstance(X_min_fake, pd.DataFrame):
      X_min_fake = X_min_fake.values

    if isinstance(y_min_fake, pd.DataFrame) or isinstance(y_min_fake, pd.Series):
      y_min_fake = y_min_fake.values.reshape(-1,1)

    if isinstance(X_maj_real, pd.DataFrame):
      X_maj_real = X_maj_real.values

    if isinstance(y_maj_real, pd.DataFrame) or isinstance(y_maj_real, pd.Series):
      y_maj_real = y_maj_real.values.reshape(-1,1)

    X_real_tensor = torch.FloatTensor(X_min_real)
    y_real_tensor = torch.FloatTensor(y_min_real)
    X_fake_tensor = torch.FloatTensor(X_min_fake)
    y_fake_tensor = torch.FloatTensor(y_min_fake)
    
    # 2. Define the GAN model layers
    generator_layers = nn.Sequential(
      get_generator_block(X_real_tensor.shape[1], 128),
      nn.Linear(128, X_real_tensor.shape[1]),
    )

    discriminator_layers = nn.Sequential(
      get_discriminator_block(X_fake_tensor.shape[1], 128),
      nn.Linear(in_features=128, out_features=y_real_tensor.shape[1]),
    )

    bce_logits_criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer_g = optim.AdamW(generator_layers.parameters(), lr=0.01)
    optimizer_d = optim.AdamW(discriminator_layers.parameters(), lr=0.001)
    
    # 3. Train GAN
    disc_loss = []
    gen_loss = []
    for epoch in tqdm(range(epochs)):
      
      # Train Discriminator
      optimizer_d.zero_grad()

      # Pass real data through discriminator
      real_preds = discriminator_layers(X_real_tensor)
      loss_real = bce_logits_criterion(real_preds, y_real_tensor)
     
      # Generate fake data 
      fake_data = generator_layers(X_fake_tensor)
      
      # Pass fake data through discriminator    
      fake_preds = discriminator_layers(fake_data)
      fake_labels = torch.zeros(len(X_fake_tensor), 1)
      loss_fake = bce_logits_criterion(fake_preds, fake_labels)

      loss_d = (loss_real + loss_fake)/2
      loss_d.backward()
      optimizer_d.step()

      # Train Generator
      optimizer_g.zero_grad()
      
      # Generate fake images
      fake_data = generator_layers(X_fake_tensor)
      
      # Try to fool the discriminator
      fake_preds = discriminator_layers(fake_data)
      loss_g = bce_logits_criterion(fake_preds, y_fake_tensor)
      loss_g.backward()
      optimizer_g.step()

      disc_loss.append(loss_d.item())
      gen_loss.append(loss_g.item())

      if early_stop(gen_loss):
        print(f"met early stopping criteria with min gen loss: {gen_loss[-1]}")
        break

    # 4. Generate synthetic samples
    with torch.no_grad():
      synthetic_samples = generator_layers(X_fake_tensor).numpy()
      # print(calculate_fid(X_min_real, synthetic_samples))

    # Append the synthetic samples to original X and adjust y accordingly
    X_resampled = np.vstack([X_min_real, synthetic_samples, X_maj_real])
    y_resampled = np.hstack([y_min_real.flatten(), y_min_fake.flatten(), y_maj_real.flatten()])

    print(pd.DataFrame(zip(disc_loss, gen_loss), columns = ['disc_loss', 'gen_loss']).plot())

    return shuffle_in_unison(X_resampled, y_resampled)

# COMMAND ----------

X_train_oversampled = X_train_SMOTE[X_train.shape[0]:]
y_train_oversampled = y_train_SMOTE[y_train.shape[0]:]

X_train_resampled, y_train_resampled = apply_smote_gan(X_min_train, y_min_train, 
                                                      X_train_oversampled, y_train_oversampled,
                                                      X_maj_train, y_maj_train, 100)

# COMMAND ----------

X_train_resampled.shape, y_train_resampled.shape, y_train_resampled.mean()

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

# COMMAND ----------

lr = LogisticRegression(C=0.01, solver='saga', penalty='elasticnet', l1_ratio=0.5)
lr.fit(X_train_resampled, y_train_resampled)

# COMMAND ----------

y_pred_train = np.around(lr.predict_proba(X_train.values)[:, 1]*1000, decimals=2)
y_pred_test = np.around(lr.predict_proba(X_test.values)[:, 1]*1000, decimals=2)

y_pred_train.mean(), y_pred_test.mean(), y_pred_test[ftt_test].mean(), y_pred_test[~ftt_test].mean()

# COMMAND ----------

from sklearn.metrics import roc_curve, auc, balanced_accuracy_score

# COMMAND ----------

fpr, tpr, thresholds = roc_curve(y_train, y_pred_train)
print(auc(fpr, tpr))
fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)
print(auc(fpr, tpr))

# COMMAND ----------

fpr, tpr, thresholds = roc_curve(y_train[ftt_train], y_pred_train[ftt_train])
print(auc(fpr, tpr))
fpr, tpr, thresholds = roc_curve(y_train[~ftt_train], y_pred_train[~ftt_train])
print(auc(fpr, tpr))

# COMMAND ----------

fpr, tpr, thresholds = roc_curve(y_test[ftt_test], y_pred_test[ftt_test])
print(auc(fpr, tpr))
fpr, tpr, thresholds = roc_curve(y_test[~ftt_test], y_pred_test[~ftt_test])
print(auc(fpr, tpr))

# COMMAND ----------

ftt_threshold, ntt_threshold = np.percentile(y_pred_train[ftt_train], q=95.0), np.percentile(y_pred_train[~ftt_train], q=95.0)
ftt_threshold, ntt_threshold

# COMMAND ----------

print(balanced_accuracy_score(y_train[ftt_train], np.where(y_pred_train[ftt_train] >= ftt_threshold, 1, 0)),
      balanced_accuracy_score(y_train[~ftt_train], np.where(y_pred_train[~ftt_train] >= ntt_threshold, 1, 0)))

# COMMAND ----------

print(balanced_accuracy_score(y_test[ftt_test], np.where(y_pred_test[ftt_test] >= ftt_threshold, 1, 0)), 
      balanced_accuracy_score(y_test[~ftt_test], np.where(y_pred_test[~ftt_test] >= ntt_threshold, 1, 0)))

# COMMAND ----------


