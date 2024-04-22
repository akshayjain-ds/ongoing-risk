# Databricks notebook source
# MAGIC %run ./set_up

# COMMAND ----------

# MAGIC %pip install tensorflow-cpu

# COMMAND ----------

# MAGIC %run ./train_metadata

# COMMAND ----------

# MAGIC %run ./pre-processing

# COMMAND ----------

# %run ./functions

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
import gc
import tensorflow
import tensorflow.keras as keras
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
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

markers = []
for id, series in features.groupby([id1])[target_b]:
  
  marker = []
  counter = 0
  
  for b in series.to_list():

    if b == 0:

      if counter == 0:
        marker.append(True)
      else:
        marker.append(False)
    
    else:

      if counter == 0:
        marker.append(True)
        counter +=1
      else:
        marker.append(False)    
  
  markers += marker
  
features = features[markers]
features.shape

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

train_dataset = features[features[id1].isin(train_ids.tolist())]
test_dataset = features[features[id1].isin(test_ids.tolist())]

train_dataset.sort_values(by=[id1, timestamp], inplace=True)
test_dataset.sort_values(by=[id1, timestamp], inplace=True)

train_dataset.shape, test_dataset.shape


# COMMAND ----------

print(np.average(train_dataset[target_b]),
      np.average(test_dataset[target_b]))

# COMMAND ----------

train_dataset[id1].nunique(), test_dataset[id1].nunique()

# COMMAND ----------

train_dataset.shape[0]/train_dataset[id1].nunique(), test_dataset.shape[0]/test_dataset[id1].nunique()

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

def data_generator(data: pd.DataFrame, 
                   timesteps: int = 5, 
                   id: str = id1, 
                   feature_list: list = feature_list):

  rows = data[id].nunique()
  X = np.empty((rows, timesteps, len(feature_list)))
  y = np.empty((rows, timesteps))
  y2 = np.empty((rows, timesteps))
  is_ftt = np.empty((rows, timesteps), dtype=bool)
  not_padded_ind = np.empty((rows, timesteps), dtype=bool)

  def padseq_mat(mat, size = timesteps):
    
    t = size - len(mat)
    
    if t > 0:
      mat = np.pad(mat, pad_width=((t, 0), (0, 0)), mode='constant')
    elif t < 0:
      mat = mat[-size:, :]
    else:
      mat = mat
  
    return mat
  
  def padseq_vec(vec, size = timesteps):
    
    t = size - len(vec)

    if t > 0:
      vec = np.pad(vec, pad_width=((t, 0),), mode='constant')
    elif t < 0:
      vec = vec[-size:]
    else:
      vec = vec

    return vec
  
  def not_padded_vec(vec, size = timesteps):
    
    t = size - len(vec)
    not_padded = np.ones(len(vec))

    if t > 0:
      not_padded = np.pad(not_padded, pad_width=((t, 0),), mode='constant')
    elif t < 0:
      not_padded[:np.abs(t)] = 0
      not_padded = not_padded[-size:]
    else:
      not_padded = not_padded

    return not_padded

  for i, (index, df) in tqdm(enumerate(data.groupby(id))):

    X[i] = padseq_mat(df[feature_list].values)
    y[i] = padseq_vec(df[target_b].values)
    y2[i] = padseq_vec(df[target_c].values)
    is_ftt[i] = padseq_vec((df['is_ftt'] > 0)*1).astype(bool)
    not_padded_ind[i] = not_padded_vec(df[target_b].values).astype(bool)

  return (X, y, y2, is_ftt, not_padded_ind)


# COMMAND ----------

timesteps = 5
X_train, y_train, y2_train, ftt_train, not_padded_train = data_generator(train_dataset, timesteps)
X_test, y_test, y2_test, ftt_test, not_padded_test = data_generator(test_dataset, timesteps)
print(X_train.shape, y_train.shape, y2_train.shape, not_padded_train.shape,
      X_test.shape, y_test.shape, y2_test.shape, not_padded_test.shape)

# COMMAND ----------

y_train[not_padded_train].shape[0]/(y_train.shape[0]*y_train.shape[1])

# COMMAND ----------

y_test[not_padded_test].shape[0]/(y_test.shape[0]*y_test.shape[1])

# COMMAND ----------

print(y_train[not_padded_train].flatten().mean(), 
      y_test[not_padded_test].flatten().mean())

# COMMAND ----------

print(y_train[not_padded_train * ftt_train].flatten().mean(), 
      y_test[not_padded_test * ftt_test].flatten().mean())

# COMMAND ----------

print(y_train[not_padded_train * ~ftt_train].flatten().mean(), 
      y_test[not_padded_test * ~ftt_test].flatten().mean())

# COMMAND ----------

model = keras.Sequential()
model.add(keras.layers.Input((timesteps, len(feature_list))))

model.add(keras.layers.LSTM(len(feature_list), activation='relu',
          dropout=0.2, recurrent_dropout=0.2, seed=seed, return_sequences=True))

model.add(keras.layers.LSTM(len(feature_list)//2, activation='relu',
          dropout=0.2, recurrent_dropout=0.2, seed=seed, return_sequences=True))

model.add(keras.layers.LSTM(len(feature_list)//4, activation='relu',
          dropout=0.2, recurrent_dropout=0.2, seed=seed))

model.add(keras.layers.Dense(timesteps, activation="sigmoid"))
print(model.summary())

# COMMAND ----------

model.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[keras.metrics.AUC(), keras.metrics.FalsePositives],
)	

# COMMAND ----------

counts = np.bincount(y_train[:, -1].astype(int))
print(
    "Number of positive samples in training data: {} ({:.2f}% of total)".format(
        counts[1], 100 * float(counts[1]) / len(y_train[:, -1])
    )
)

weight_for_0 = 1000 / counts[0]
weight_for_1 = 1000 / counts[1]

class_weight = {0: weight_for_0, 1: weight_for_1}
class_weight

# COMMAND ----------

my_callbacks = [
    keras.callbacks.EarlyStopping(patience=3),
]

model.fit(
    X_train, y_train, 
    # class_weight=class_weight,
    validation_data=(X_test, y_test), 
    batch_size=1000, epochs=100, 
    callbacks=my_callbacks, shuffle=True,
)

# COMMAND ----------

y_pred_train = np.around(model.predict(X_train)*1000, 2)
y_pred_test = np.around(model.predict(X_test)*1000, 2)
y_pred_train.shape, y_pred_test.shape

# COMMAND ----------

print(
  y_pred_train[not_padded_train].mean(),
  y_pred_test[not_padded_test].mean()
  )

# COMMAND ----------

print(
  y_pred_train[not_padded_train * y_train.astype(bool)].mean(),
  y_pred_test[not_padded_test * y_test.astype(bool)].mean()
  )

# COMMAND ----------

print(
  y_pred_train[not_padded_train * ~y_train.astype(bool)].mean(),
  y_pred_test[not_padded_test * ~y_test.astype(bool)].mean()
  )

# COMMAND ----------

print(
  y_pred_train[not_padded_train].mean(),
  y_pred_test[not_padded_test].mean()
  )

# COMMAND ----------

print(
  y_pred_train[not_padded_train * ftt_train].mean(), 
  y_pred_test[not_padded_test * ftt_test].mean()
  )

# COMMAND ----------

print(
  y_pred_train[not_padded_train * ~ftt_train].mean(), 
  y_pred_test[not_padded_test * ~ftt_test].mean()
  )

# COMMAND ----------

fpr, tpr, thresholds = roc_curve(y_train[not_padded_train], 
                                 y_pred_train[not_padded_train])
print(auc(fpr, tpr))
fpr, tpr, thresholds = roc_curve(y_test[not_padded_test], 
                                 y_pred_test[not_padded_test])
print(auc(fpr, tpr))

# COMMAND ----------

fpr, tpr, thresholds = roc_curve(y_train[not_padded_train * ftt_train],  
                                 y_pred_train[not_padded_train * ftt_train])
print(auc(fpr, tpr))
fpr, tpr, thresholds = roc_curve(y_test[not_padded_test * ftt_test], 
                                 y_pred_test[not_padded_test * ftt_test])
print(auc(fpr, tpr))

# COMMAND ----------

fpr, tpr, thresholds = roc_curve(y_train[not_padded_train * ~ftt_train], 
                                 y_pred_train[not_padded_train * ~ftt_train])
print(auc(fpr, tpr))
fpr, tpr, thresholds = roc_curve(y_test[not_padded_test * ~ftt_test], 
                                 y_pred_test[not_padded_test * ~ftt_test])
print(auc(fpr, tpr))

# COMMAND ----------

lift_report(y_train[not_padded_train], y_pred_train[not_padded_train], n=10)

# COMMAND ----------

lift_report(y_test[not_padded_test], y_pred_test[not_padded_test], n=10)

# COMMAND ----------


