# Databricks notebook source
# MAGIC %run ./set_up

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

# MAGIC %run ./feature_transformation

# COMMAND ----------

import scipy as sp
import pandas as pd
import json
from datetime import timedelta
import tecton
from optbinning import OptimalBinning
import gc
import category_encoders as ce
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_curve, auc, det_curve, balanced_accuracy_score
from sklearn.pipeline import (make_pipeline, Pipeline)
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import (StandardScaler, OneHotEncoder, MinMaxScaler)
from sklearn.feature_selection import VarianceThreshold
gc.enable()

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

day_of_month = 'fixed_day'
features = pd.read_csv(f"/Workspace/Shared/Decisioning/Strawberries/ongoing-ntt_mp_risk-v1/artefacts/{'ntt_ftt' if include_ftt else 'ntt'}_{day_of_month}_raw_features_{train_start_date}_{val_end_date}.csv.gz", 
                 dtype={id1: str, id2: str})
features = features.assign(

  company_created_on = lambda col: pd.to_datetime(col[created_on]),
  timestamp = lambda col: pd.to_datetime(col[timestamp]),
  approved_at = lambda col:  pd.to_datetime(col['approved_at'])

)     
features = features[~pd.isnull(pd.to_numeric(features[id1], errors='coerce'))]

def sic_converter(x):
  
  if str(x) != 'nan':
    try:
      return str(int(x)).zfill(5)
    except ValueError:
      return np.NaN
  else:
    return np.NaN

for f in [f for f in features.columns if f.__contains__('sic')]:
  features[f] = features[f].apply(lambda x: sic_converter(x))      
features.shape

# COMMAND ----------

features.drop(columns=[target_b, target_c, target_d], inplace=True)

target_b, target_c, target_d = 'is_app_fraud_45d', 'app_fraud_amount_45d', 'days_to_fraud_45d'

fixed_day_df = pd.read_csv(f"/Workspace/Shared/Decisioning/Strawberries/ongoing-ntt_mp_risk-v1/artefacts/ntt_ftt_fixed_day_base_2022-01-01_2023-12-31.csv.gz", memory_map=True, dtype={id1: str})
fixed_day_df = fixed_day_df.assign(
  timestamp = lambda col: pd.to_datetime(col[timestamp]),
)
features = features.merge(fixed_day_df[[id1, timestamp, target_b, target_c, target_d]], on=[id1, timestamp])
features.shape

# COMMAND ----------

del fixed_day_df
gc.collect()

# COMMAND ----------

train_dataset = features[(pd.to_datetime(features[timestamp]) >= pd.to_datetime(train_start_date)) & 
                   (pd.to_datetime(features[timestamp]) <= pd.to_datetime(train_end_date))].sample(frac=1.0, random_state=seed)
test_dataset = features[(pd.to_datetime(features[timestamp]) >= pd.to_datetime(test_start_date)) & 
                   (pd.to_datetime(features[timestamp]) <= pd.to_datetime(test_end_date))].sample(frac=1.0, random_state=seed)
val_dataset = features[(pd.to_datetime(features[timestamp]) >= pd.to_datetime(val_start_date)) & 
                   (pd.to_datetime(features[timestamp]) <= pd.to_datetime(val_end_date))].sample(frac=1.0, random_state=seed)

for data in [train_dataset, test_dataset, val_dataset]:
  company_counts = data.groupby(id1).size()
  mean_cc = company_counts.mean()
  data['weight'] = data[id1].apply(lambda x: mean_cc / company_counts.loc[x])
  
train_labels = train_dataset[[timestamp, target_b, target_d, target_c]]
test_labels = test_dataset[[timestamp, target_b, target_d, target_c]]
val_labels = val_dataset[[timestamp, target_b, target_d, target_c]]

train_labels[target_b] = train_labels[target_b].astype(bool)
test_labels[target_b] = test_labels[target_b].astype(bool)
val_labels[target_b] = val_labels[target_b].astype(bool)

ftt_train = train_dataset['is_ftt'].astype(bool)
ftt_test = test_dataset['is_ftt'].astype(bool)
ftt_val = val_dataset['is_ftt'].astype(bool)

w_train = train_dataset['weight']
w_test = test_dataset['weight']
w_val = val_dataset['weight']

scaler = MinMaxScaler()
w2_train = pd.Series(
  scaler.fit_transform(
    w_train.values.reshape(-1,1)
    + (
      scaler.fit_transform(train_dataset[[target_c]].fillna(0))
      # * scaler.fit_transform(train_dataset[[target_d]].fillna(0))
    )
    ).flatten(),
  index=w_train.index) * calculate_weight_decay(train_dataset, timestamp) #.values.reshape(-1,1)
w2_train = w2_train * (w2_train.shape[0]/w2_train.sum())

train_dataset.shape, test_dataset.shape, val_dataset.shape

# COMMAND ----------

train_dataset[timestamp].nunique(), test_dataset[timestamp].nunique(), val_dataset[timestamp].nunique()

# COMMAND ----------

print(np.average(train_labels[target_b]),
      np.average(train_labels[target_b], weights=w_train), 
      np.average(train_labels[target_b], weights=w2_train), 
      np.average(test_labels[target_b], weights=w_test), 
      np.average(val_labels[target_b], weights=w_val))

# COMMAND ----------

print(np.average(train_labels[ftt_train][target_b]),
      np.average(train_labels[ftt_train][target_b], weights=w_train[ftt_train]), 
      np.average(train_labels[ftt_train][target_b], weights=w2_train[ftt_train]), 
      np.average(test_labels[ftt_test][target_b], weights=w_test[ftt_test]), 
      np.average(val_labels[ftt_val][target_b], weights=w_val[ftt_val]))

# COMMAND ----------

print(np.average(train_labels[~ftt_train][target_b]),
      np.average(train_labels[~ftt_train][target_b], weights=w_train[~ftt_train]), 
      np.average(train_labels[~ftt_train][target_b], weights=w2_train[~ftt_train]), 
      np.average(test_labels[~ftt_test][target_b], weights=w_test[~ftt_test]), 
      np.average(val_labels[~ftt_val][target_b], weights=w_val[~ftt_val]))

# COMMAND ----------

train_dataset[id1].nunique(), train_dataset['weight'].sum(), w2_train.sum(), test_dataset[id1].nunique(), test_dataset['weight'].sum(), val_dataset[id1].nunique(), val_dataset['weight'].sum()

# COMMAND ----------

feature_list = [
 'days_on_books',
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
 'section_description',
 'company_age_at_timestamp',
#  'attribution_marketing_campaign',
 'attribution_marketing_channel',
 'receipt_uploaded_before_timestamp',
 'receipt_match_before_timestamp',
 'first_invoice_before_timestamp',
 'invoice_matched_before_timestamp',
 'invoice_chased_before_timestamp',
 'company_created_on',
 'company_postcode',
 'company_structurelevelwise_1',
 'company_directors_count',
 'applicant_nationality_0',
 'applicant_nationality_1',
 'applicant_nationality_2',
 'applicant_postcode',
 'applicant_idcountry_issue',
 'applicant_id_type',
 'applicant_email_numeric',
 'applicant_email_domain',
 'applicant_age_at_completion',
 'days_to_approval',
 'applicant_device_type',
 'company_icc',
 'applicant_years_to_id_expiry',
 'company_type',
 'is_restricted_keyword_present',
#  'manual_approval_triggers',
 'company_is_registered',
 'company_sic_0',
 'company_sic_1',
 'company_sic_2',
 'company_sic_3'
 ]
len(feature_list)

# COMMAND ----------

pd.isnull(train_dataset[feature_list]).sum()/train_dataset.shape[0]

# COMMAND ----------

from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, accuracy_score, balanced_accuracy_score

feature = 'is_ftt'
print(np.average(train_dataset[target_b], weights=train_dataset['weight']))
print(precision_recall_fscore_support(train_dataset[target_b], 
                                train_dataset[feature],
                                 sample_weight=train_dataset['weight'],
                                 average='binary')[:3])
print(matthews_corrcoef(train_dataset[target_b], 
                        train_dataset[feature],
                        sample_weight=train_dataset['weight']))
print(balanced_accuracy_score(train_dataset[target_b], 
                        train_dataset[feature],
                        sample_weight=train_dataset['weight']))

# COMMAND ----------

class CategoryTransformer(BaseEstimator, TransformerMixin):
  
  @staticmethod
  def create_var(var_name, var_value):
    globals()[var_name] = var_value

  def __init__(self, 
                estimator: object,
                category_features: list):
      
      self.category_features = category_features
      self.estimator = estimator
      self.encoders = {}
      self.sic_count = 0
      self.nationality_count = 0
      self.fitted = False

  def fit(self, X: pd.DataFrame, y=pd.Series):

    for input_feature in self.category_features:

      if input_feature.__contains__("sic"):

        orig_input_feature = "_".join(input_feature.split("_")[:-1])
        self.sic_count +=1
      
        if input_feature.__contains__("sic_0"):
        
          mest_input = copy.deepcopy(self.estimator)
          mest_input.feature_encoder.cols=[orig_input_feature]
          mest_input.fit(X[input_feature].rename(orig_input_feature), y)
          self.encoders[orig_input_feature] = mest_input

      elif input_feature.__contains__("nationality"):
        self.nationality_count +=1

        orig_input_feature = "_".join(input_feature.split("_")[:-1])

        if input_feature.__contains__("nationality_0"):

          mest_input = copy.deepcopy(self.estimator)
          mest_input.feature_encoder.cols=[orig_input_feature]
          mest_input.fit(X[input_feature].rename(orig_input_feature), y)
          self.encoders[orig_input_feature] = mest_input

      else:
        
        mest_input = copy.deepcopy(self.estimator)
        mest_input.feature_encoder.cols=[input_feature]
        CategoryTransformer.create_var(f"mest_{input_feature}", mest_input)
        eval(f"mest_{input_feature}").fit(X[f'{input_feature}'], y)
        self.encoders[f'{input_feature}'] = eval(f"mest_{input_feature}")

    self.fitted = True
    return self
  
  def transform(self, X: pd.DataFrame):

    if self.fitted:
      
      for key, encoder in self.encoders.items():

        if key in ['company_sic']:

          cols = [f'{key}_{i}' for i in range(self.sic_count)]
          for i, col in enumerate(cols):
            X[f'{col}_encoded'] = encoder.transform(X[col].rename(key))
          X[f"{key}_encoded"] = X[[f'{col}_encoded' for col in cols]].max(axis=1)
        
        elif key in ['applicant_nationality']:

          cols = [f'{key}_{i}' for i in range(self.nationality_count)]
          for i, col in enumerate(cols):
            X[f'{col}_encoded'] = encoder.transform(X[col].rename(f"{key}"))
          X[f"{key}_encoded"] = X[[f'{col}_encoded' for col in cols]].max(axis=1)

        else:
          X[f"{key}_encoded"] = encoder.transform(X[f"{key}"])
      
      return X

    else:
      
      raise(NotFittedError("This CategoryTransformer instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."))

# COMMAND ----------

# learn continuous target encodings based on M-EST technique for categorical features with high cardinality

cv = 5
avg_company_repetition = train_dataset.shape[0]/train_dataset[id1].nunique()
leaf_size = int(train_dataset.shape[0]*0.01/avg_company_repetition)
print(leaf_size)

p = np.average(train_labels[target_b], weights=w2_train)
std = np.around(np.sqrt(p*(1-p)), decimals=2)
print(std)

m_est = ce.wrapper.NestedCVWrapper(ce.MEstimateEncoder(sigma=std, randomized=True,
                                                       handle_missing='return_nan', handle_unknown='return_nan'),
                                   cv=cv, 
                                   random_state=seed)




category_transformer = CategoryTransformer(
  estimator=m_est,
  category_features=train_dataset[feature_list].select_dtypes('O').columns.tolist()
)

category_transformer.fit(train_dataset, train_labels[target_b])
category_transformer.encoders

# COMMAND ----------

class NOBTransformer(BaseEstimator, TransformerMixin):

  def __init__(self,
               new_feature: str,
               encoded_features: list):
      
      self.new_feature = new_feature
      self.encoded_features = encoded_features

  def fit(self, X: pd.DataFrame, y=pd.Series):

    return self
  
  def transform(self, X: pd.DataFrame):
      
    X[self.new_feature] = X[self.encoded_features].max(axis=1)
    X.drop(columns=self.encoded_features, inplace=True)
    
    return X
  

# COMMAND ----------

nob_transformer = NOBTransformer(new_feature = "company_nob_encoded", 
                                 encoded_features = ['company_icc_encoded', 
                                                     'company_sic_encoded', 'section_description_encoded'])

for data in [train_dataset, test_dataset, val_dataset]:
  data = category_transformer.transform(data)
  data = nob_transformer.transform(data)

# COMMAND ----------

class AddNoiseTransformer(BaseEstimator, TransformerMixin):

  def __init__(self,
               features: list,
               std_scale_factors: list,
               random_state: int = seed):
      
      self.features = features
      self.std_scale_factors = std_scale_factors
      self.metrics = {}
      self.fitted=False
      self.rng = np.random.default_rng(random_state)

  def fit(self, X: pd.DataFrame, y=pd.Series, w=pd.Series):

    for feature, scale in zip(self.features, self.std_scale_factors):

      mu, sig = weighted_avg_and_std(X[~pd.isnull(X[feature])][feature], 
                                    w[~pd.isnull(X[feature])])
      
      self.metrics[feature] = {"mu": mu, "sig": sig, "std_scale": scale}
      print(feature, mu, sig, get_iv_class(y, X[f'{feature}'], f'{feature}')[-1])

      noisy_feature = X[feature] + self.rng.normal(
        0,
        sig/scale,
        X[feature].shape) #white noise
      mu, sig = weighted_avg_and_std(noisy_feature[~pd.isnull(noisy_feature)], 
                                    w[~pd.isnull(noisy_feature)])
      print(feature, mu, sig, get_iv_class(y, noisy_feature, f'{feature}')[-1])

    self.fitted=True
    return self
  
  def transform(self, X: pd.DataFrame):
    
    if self.fitted:

      for feature, value in self.metrics.items():
        
        mu, sig, scale = value.get("mu"), value.get("sig"), value.get("std_scale")

        X[f'{feature}'] = X[feature] + self.rng.normal(
          0,
          sig/scale,
          X[feature].shape) #white noise
      
      return X
    
    else:

      raise(NotFittedError("This AddNoiseTransformer instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."))

# COMMAND ----------

add_noise_transformer = AddNoiseTransformer(
  features = ['company_nob_encoded', 
              'applicant_idcountry_issue_encoded', 
              'applicant_nationality_encoded'
              ],
  std_scale_factors = [3, 3, 3],
  random_state = seed
)
add_noise_transformer.fit(train_dataset, train_labels[target_b], w2_train)

# COMMAND ----------

train_dataset = add_noise_transformer.transform(train_dataset)

# COMMAND ----------

class WOETransformer(BaseEstimator, TransformerMixin):
  
  @staticmethod
  def create_var(var_name, var_value):
    globals()[var_name] = var_value

  def __init__(self, 
                estimator: object,
                features: list):
      
      self.features = features
      self.estimator = estimator
      self.encoders = {}
      self.fitted=False

  
  def fit(self, X: pd.DataFrame, y=pd.Series):

    for input_feature in self.features:

      WOETransformer.create_var(f"optb_{input_feature}", 
                                self.estimator(name=f"{input_feature}", 
                                               dtype="numerical", solver="cp",
                                               max_n_prebins=33, min_prebin_size=0.033))

      eval(f"optb_{input_feature}").fit(X[f"{input_feature}"], y)
      
      self.encoders[f"{input_feature}"] = eval(f"optb_{input_feature}")

    self.fitted = True
    return self
  
  def transform(self, X: pd.DataFrame):

    if self.fitted:

      for key, value in self.encoders.items():

        X[f"{key}{'' if key.endswith('_encoded') else '_encoded'}"] = value.transform(X[f"{key}"],
                                                                                metric="woe",metric_missing='empirical')
        
      return X

    else:

      raise(NotFittedError("This WOETransformer instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."))
    

# COMMAND ----------

woe_feature_list = [col for col in list(map(lambda col: f"{col}_encoded", list(category_transformer.encoders))) if col not in ['company_icc_encoded', 'company_sic_encoded', 'section_description_encoded']] + ['company_nob_encoded'] + ['company_age_at_timestamp', 'applicant_years_to_id_expiry', 'days_to_transact']
print(len(woe_feature_list))
print(woe_feature_list)

# COMMAND ----------

num_feature_list = list(set(train_dataset[train_dataset[feature_list].select_dtypes(np.number).columns.tolist()]).difference(set(woe_feature_list)))
print(len(num_feature_list))
print(num_feature_list)

# COMMAND ----------

woe_transformer = WOETransformer(
  estimator=OptimalBinning,
  features=woe_feature_list
)

woe_transformer.fit(train_dataset, train_labels[target_b])
woe_transformer.encoders

# COMMAND ----------

woe_feature_list

# COMMAND ----------

pd.isnull(train_dataset[num_feature_list]).sum()

# COMMAND ----------

for key, value in woe_transformer.encoders.items():
  print(value.binning_table.build())
  print(value.binning_table.plot(metric="woe", add_missing=True))

# COMMAND ----------

catgory_pipeline = make_pipeline(category_transformer, nob_transformer, woe_transformer)
numeric_pipeline = make_pipeline(SimpleImputer(strategy='constant', fill_value=0), StandardScaler())
numeric_pipeline.fit(train_dataset[num_feature_list])

# COMMAND ----------

for data in [train_dataset, test_dataset, val_dataset]:
  data = woe_transformer.transform(data)
  data[num_feature_list] = numeric_pipeline.transform(data[num_feature_list])

# COMMAND ----------

feature_list_encoded = [f"{col}{'' if col.endswith('_encoded') else '_encoded'}" for col in list(woe_transformer.encoders)] + num_feature_list
len(feature_list_encoded)

# COMMAND ----------

pd.isnull(train_dataset[feature_list_encoded]).sum()

# COMMAND ----------

iv_df = pd.DataFrame()
for f in feature_list_encoded:
  iv_df = iv_df.append([get_iv_class(train_dataset[target_b], train_dataset[f], f)])
iv_df.columns=['feature', 'iv']
iv_df['power'] = iv_df['iv'].apply(lambda x: 'suspecious' if x> 0.5 else 
                                    ('strong' if x>0.3 else 
                                     ('medium' if x>0.1 else 
                                      ('weak' if x>=0.02 else 'useless'))))
iv_df.sort_values(by=['iv'], ascending=False, inplace=True)
iv_df

# COMMAND ----------

fs = FeatureSelection()
fs.fit(train_dataset[feature_list_encoded])
feature_list2 = fs.transform(train_dataset[feature_list_encoded]).columns.tolist()
len(feature_list2)

# COMMAND ----------

selection_pipeline = make_pipeline(fs)
pipeline_steps = make_pipeline(catgory_pipeline, numeric_pipeline, selection_pipeline)
feature_transformer = Pipeline([("preprocessing", pipeline_steps)])
feature_transformer

# COMMAND ----------

vifs = pd.Series(np.linalg.pinv(train_dataset.loc[:, feature_list2].corr().to_numpy()).diagonal(), index=feature_list2, name='VIF')
vifs.sort_values(ascending=False)

# COMMAND ----------

final_list = [id1, timestamp, 'is_ftt', target_b, target_c, target_d] + feature_list2
encoded_dataset = pd.concat([train_dataset[final_list], test_dataset[final_list], val_dataset[final_list]])
encoded_dataset.shape

# COMMAND ----------

encoded_dataset.to_csv(f"/Workspace/Shared/Decisioning/Strawberries/ongoing-ntt_mp_risk-v1/artefacts/{'ntt_ftt' if include_ftt else 'ntt'}_{day_of_month}_encoded_features_{train_start_date}_{val_end_date}.csv.gz", index=False, compression='gzip')
del encoded_dataset
gc.collect()

# COMMAND ----------

feature_list2.remove('cash_dep_pct')
len(feature_list2)

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1, l1_ratio = 0.5,
                             class_weight='balanced', 
                             solver='saga', 
                             penalty='elasticnet', 
                             random_state=seed,
                             verbose=0)

# COMMAND ----------

model.fit(train_dataset[feature_list2], 
          train_labels[target_b]*1, 
          sample_weight = w2_train
       )

# COMMAND ----------

coefficients = model.coef_[0]
feature_importance = pd.DataFrame({'Feature': [feat.replace("_encoded", "") for feat in feature_list2], 'Importance': np.abs(coefficients)})
feature_importance = feature_importance.sort_values('Importance', ascending=True)
feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 12), legend=False)

# COMMAND ----------

cal_model = CalibratedClassifierCV(model, cv='prefit')
cal_model.fit(train_dataset[feature_list2], 
              train_labels[target_b], 
              sample_weight = w_train)

# COMMAND ----------

y_pred_train = np.around(cal_model.predict_proba(train_dataset[feature_list2])[:, 1]*1000, 2)
y_pred_test = np.around(cal_model.predict_proba(test_dataset[feature_list2])[:, 1]*1000, 2)
y_pred_val = np.around(cal_model.predict_proba(val_dataset[feature_list2])[:, 1]*1000, 2)
y_pred_train.mean(), y_pred_test.mean(), y_pred_val.mean()

# COMMAND ----------

y_pred_train[ftt_train].mean(), y_pred_test[ftt_test].mean(), y_pred_val[ftt_val].mean()

# COMMAND ----------

y_pred_train[~ftt_train].mean(), y_pred_test[~ftt_test].mean(), y_pred_val[~ftt_val].mean()

# COMMAND ----------

print("train")
fpr, tpr, thresholds = roc_curve(train_dataset[target_b], y_pred_train, 
                                 sample_weight=w_train
                                 )
print(auc(fpr, tpr))

AUC = []
print("test")
for t in sorted(test_dataset[timestamp].unique()):
  t_index = test_dataset[timestamp]==t
  fpr, tpr, thresholds = roc_curve(test_labels[t_index][target_b], y_pred_test[t_index])
  AUC.append(np.around(auc(fpr, tpr), decimals=2))
  plot_roc_auc(fpr, tpr, 'Test_appf', 'g', False)

print("val")
for t in sorted(val_dataset[timestamp].unique()):
  t_index = val_dataset[timestamp]==t
  fpr, tpr, thresholds = roc_curve(val_labels[t_index][target_b], y_pred_val[t_index])
  AUC.append(np.around(auc(fpr, tpr), decimals=2))
  plot_roc_auc(fpr, tpr, 'Val_appf', 'b', False)

# COMMAND ----------

np.average(AUC), np.std(AUC)
