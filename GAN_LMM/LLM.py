# Databricks notebook source
# MAGIC %pip install pycountry

# COMMAND ----------

# MAGIC %run ./utils_llm

# COMMAND ----------

# MAGIC %run ../train_metadata

# COMMAND ----------

import re, pycountry
_RE_COMBINE_WHITESPACE = re.compile(r"(?a:\s+)")
_RE_STRIP_WHITESPACE = re.compile(r"(?a:^\s+|\s+$)")
from tqdm.auto import tqdm
tqdm.pandas()
from datasets import Dataset
from sklearn.utils import class_weight
import torch
from transformers import DataCollatorWithPadding, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback, IntervalStrategy, pipeline, EvalPrediction
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import evaluate
import numpy as np
import scipy as sp
import os, shutil, glob
gc.enable()


# COMMAND ----------

features = pd.read_csv('/dbfs/dbfs/Shared/Decisioning/Strawberries/App_fraud/ntt_ftt_random_day_raw_features_2022-01-01_2023-12-31.csv.gz',
            dtype={id1: str}, memory_map=True)
features = features.assign(
  timestamp = lambda col: pd.to_datetime(col[timestamp])
  )
features = features[~features['applicant_postcode'].isna()]

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

features = features.assign(

  company_icc = lambda col: col['company_icc']
  .apply(lambda x: x.replace('category.', ''))
  .apply(lambda x: re.sub('[^a-zA-Z0-9 \n\.]', ' ', x))
  .apply(lambda x: _RE_COMBINE_WHITESPACE.sub(' ', x))
  .apply(lambda x: _RE_STRIP_WHITESPACE.sub('', x)),

  applicant_id_type = lambda col: col['applicant_id_type']
  .apply(lambda x: re.sub('[^a-zA-Z0-9 \n\.]', ' ', str(x).replace("Licence", "License"))),

  company_type = lambda col: col['company_type']
  .apply(lambda x: re.sub('[^a-zA-Z0-9 \n\.]', ' ', x)),

  applicant_idcountry_issue = lambda col: col['applicant_idcountry_issue'].fillna('GB')
  .apply(lambda x: pycountry.countries.get(alpha_2=x).name),

  fastpmt_beneficiaries = lambda col: col['fastpmt_beneficiaries'].fillna(0),
  fastpmt_benefactors = lambda col: col['fastpmt_benefactors'].fillna(0),
  ddebit_beneficiaries = lambda col: col['ddebit_beneficiaries'].fillna(0),

  avg_deposit = lambda col: col['avg_deposit'].fillna(0),
)      
features.shape

# COMMAND ----------

features = pd.concat([features[features.is_app_fraud_45d==1], 
                      features[features.is_app_fraud_45d==0].sample(frac=0.03, random_state=seed)]).sample(frac=1.0, random_state=seed)
gc.collect()
features.shape

# COMMAND ----------

import math

def concatenate_text(x):
  
  if math.isnan(x['company_is_registered']):
      ic = f"the account is of type {x['company_type']} and is not registered. "
  else:
    ic = f"the account is of type {x['company_type']} and registered with the {x['company_icc']} industry classification. "
  
  if math.isnan(x['company_age_at_timestamp']):
    age = f"The age of the applicant at onboarding is {int(x['applicant_age_at_completion'])} years, and the age of the business is unknown. "
  else:
    age = f"The age of the applicant at onboarding is {int(x['applicant_age_at_completion'])} years, and the age of the business is {int(x['company_age_at_timestamp'])} months. "

  if x['is_restricted_keyword_present']==1:
    c_name = f"It has high risk of fraudulent activity due to the presence of restricted keywords in its trading name; "
  else:
    c_name = f"has low or medium risk of fraudulent activity due to the absence of restricted keywords in its trading name; "

  if math.isnan(x['days_to_transact']):
    ntt_ftt = f"has not done its first transaction till now since approval; "
  else:
    ntt_ftt = f"has done its first transaction after {int(x['days_to_transact'])} days since approval; "

  if math.isnan(x['applicant_years_to_id_expiry']):
    expiry = f"valid for the next unknown years and "
  else:
    expiry = f"valid for the next {int(x['applicant_years_to_id_expiry'])} years and "

  full_text = (
    f"The member segment is '{'First Transaction to Tide' if x['is_ftt'] else 'New to Tide'}'; ",
    f"{ic}",
    f"{age}",
    f"{c_name}",
    f"{ntt_ftt}",
    f"has taken {int(x['days_to_approval'])} days to get approved after completing the onboarding event; ",
    f"has completed {int(x['days_on_books'])} days on books since approval; ",
    f"and {int(x['days_remaining_as_ntt_ftt'])} days are remaining to become a tenured member. ",
    f"The email contains {'no' if ~x['applicant_email_numeric'] else 'some'} numeric characters with a domain that is {'other than standard' if x['applicant_email_domain'] == 'other' else x['applicant_email_domain']} and uses the {x['applicant_device_type']} operating system on a mobile device. ",
    f"The proof of identification is a {x['applicant_id_type'].lower()} issued by the {x['applicant_idcountry_issue']} that is {expiry}",
    f"has been residing at the {x['applicant_postcode']} outward code. ",
    f"The number of fast payment beneficiaries added is {int(x['fastpmt_beneficiaries'])}, the number of fast payment benefactors added is {int(x['fastpmt_benefactors'])} and, the number of direct debit beneficiaries added is {int(x['ddebit_beneficiaries'])}. ",
    f"The average value of deposits made by the member is {int(x['avg_deposit'])} pounds."
  )
  return ''.join(full_text).strip()

# COMMAND ----------

features['labels'] = features['is_app_fraud_45d'].astype(int)
features['text'] = features.progress_apply(lambda row: concatenate_text(row), axis=1).values

# COMMAND ----------

features['text'].iloc[0]

# COMMAND ----------

max_length = len(features['text'].iloc[0].split())
max_length = 256
max_length

# COMMAND ----------

train_index = features[timestamp] <= pd.to_datetime(train_end_date) 
test_index = features[timestamp] > pd.to_datetime(test_start_date) 
train_transformed_df_llm = features[train_index][['text', 'labels']]
test_transformed_df_llm = features[test_index][['text', 'labels']]
train_transformed_df_llm['labels'] = train_transformed_df_llm['labels']
test_transformed_df_llm['labels'] = test_transformed_df_llm['labels']
train_transformed_df_llm.shape, test_transformed_df_llm.shape

# COMMAND ----------

train_transformed_df_llm.info()

# COMMAND ----------

train_transformed_df_llm.shape, test_transformed_df_llm.shape

# COMMAND ----------

train_transformed_df_llm['labels'].mean(), test_transformed_df_llm['labels'].mean()

# COMMAND ----------

class_weights = class_weight.compute_class_weight('balanced', [0,1], train_transformed_df_llm['labels']).tolist()
class_weights

# COMMAND ----------

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def clear_memory():
  print('Using device:', device)
  #Additional Info when using cuda
  if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
  torch.cuda.empty_cache()
  gc.collect()
clear_memory()

# COMMAND ----------


# Preparing the Text for Our Model
model_cpt = 'microsoft/MiniLM-L12-H384-uncased'
pre_trained_tokenizer = AutoTokenizer.from_pretrained(model_cpt)

def tokenize_text(examples, tokenizer=pre_trained_tokenizer):

    return tokenizer(examples['text'], truncation=True, max_length=max_length, padding='max_length')

pre_trained_model = AutoModelForSequenceClassification.from_pretrained(model_cpt, 
                                                          num_labels=2, 
                                                          id2label = {0: "LEGITIMATE", 1: "SUSPICIOUS"},
                                                          label2id = {"LEGITIMATE": 0, "SUSPICIOUS": 1})
pre_trained_model.to(device)

# COMMAND ----------

batch_size=64

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_transformed_df_llm)
test_dataset = Dataset.from_pandas(test_transformed_df_llm)

train_inputs = train_dataset.map(tokenize_text, batched=True, batch_size=batch_size)
test_inputs = test_dataset.map(tokenize_text, batched=True, batch_size=batch_size)

train_inputs = train_inputs.map(lambda x: {'labels': x['labels']})
test_inputs = test_inputs.map(lambda x: {'labels': x['labels']})

data_collator = DataCollatorWithPadding(tokenizer=pre_trained_tokenizer)

# COMMAND ----------

try:
  train_inputs = train_inputs.remove_columns(["__index_level_0__", "text"])
except:
  pass
train_inputs

# COMMAND ----------

try:
  test_inputs = test_inputs.remove_columns(["__index_level_0__", "text"])
except:
  pass
test_inputs

# COMMAND ----------

pre_trained_tokenizer.decode(train_inputs.__getitem__(0).get('input_ids'))

# COMMAND ----------

model_name = 'ntt-ftt-mp-ongoing-risk'
logging_steps = len(train_inputs) // batch_size
output_dir = f'/Workspace/Shared/Decisioning/Strawberries/ongoing-ntt_mp_risk-v1/other_methods/{model_name}/model_checkpoints/'
log_dir = f'/Workspace/Shared/Decisioning/Strawberries/ongoing-ntt_mp_risk-v1/other_methods/{model_name}/logs/'
epochs=5

for filename in os.listdir(output_dir):
  file_path = os.path.join(output_dir, filename)
  try:
      if os.path.isfile(file_path) or os.path.islink(file_path):
          os.unlink(file_path)
      elif os.path.isdir(file_path):
          shutil.rmtree(file_path)
  except Exception as e:
      print('Failed to delete %s. Reason: %s' % (file_path, e))


# COMMAND ----------

cal_precision = evaluate.load("precision")
cal_recall = evaluate.load("recall")
cal_f1 = evaluate.load("f1")

def compute_metrics(eval_predictions: EvalPrediction):
    
    labels = eval_predictions.label_ids
    logits = eval_predictions.predictions
    
    predictions = np.argmax(sp.special.softmax(logits, axis=1), axis=1)

    precision = cal_precision.compute(predictions=predictions, references=labels, average='binary', pos_label=1)["precision"]
    recall = cal_recall.compute(predictions=predictions, references=labels, average='binary', pos_label=1)["recall"]
    f1 = cal_f1.compute(predictions=predictions, references=labels, average='binary', pos_label=1)["f1"]

    return {
        "precision": precision, 
        "recall": recall, 
        "f1": f1
    }

# COMMAND ----------

clear_memory()
training_args = TrainingArguments(
    # output_dir=output_dir,
    output_dir=model_name,
    lr_scheduler_type = 'cosine_with_restarts',
    learning_rate=0.001,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end = True,
    metric_for_best_model = 'eval_loss',
    # metric_for_best_model = 'eval_f1',
    logging_dir=log_dir,
    fp16=True,
    save_total_limit=3,
    seed=seed
)

# COMMAND ----------

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_inputs,
#     eval_dataset=test_inputs,
#     callbacks = [transformers.EarlyStoppingCallback(early_stopping_patience=5)]
# )

class CWTrainer(Trainer):

    def __init__(self, class_weights: torch.tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs.pop("labels")

        # forward pass
        outputs = model(**inputs)
        
        # get logits (log odds ratios) from model
        logits = outputs.get('logits')

        # compute loss with weights
        criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights, reduction='mean')
        loss = criterion(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss
    

# ongoing_trainer = Trainer(
ongoing_trainer = CWTrainer(
    class_weights = torch.tensor(class_weights, device=device),
    model=pre_trained_model,
    args=training_args,
    train_dataset=train_inputs,
    eval_dataset=test_inputs,
    # tokenizer=pre_trained_tokenizer,
    # data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
)

history = ongoing_trainer.train()
ongoing_trainer.save_model(f"{output_dir}{model_name}")

# COMMAND ----------

# Evaluate the model
results = ongoing_trainer.evaluate()
print(results)

# COMMAND ----------

y_train_predictions = ongoing_trainer.predict(train_inputs)
y_test_predictions = ongoing_trainer.predict(test_inputs)

# COMMAND ----------

y_test_predictions

# COMMAND ----------

logits = torch.tensor(y_train_predictions.predictions)
y_pred_train = torch.softmax(logits, dim=1).numpy()[:, 1]
logits = torch.tensor(y_test_predictions.predictions)
y_pred_test = torch.softmax(logits, dim=1).numpy()[:, 1]

# COMMAND ----------

y_pred_train.mean(), y_pred_test.mean()

# COMMAND ----------

from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(train_transformed_df_llm.labels, y_pred_train)
print(auc(fpr, tpr))
fpr, tpr, thresholds = roc_curve(test_transformed_df_llm.labels, y_pred_test)
print(auc(fpr, tpr))

# COMMAND ----------

# from lime.lime_text import LimeTextExplainer

# # Ensure the model is in evaluation mode and moved to CPU
# pre_trained_model.eval()
# pre_trained_model.to('cpu')

# def predictor(texts):
#   inputs = pre_trained_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
#   with torch.no_grad():
#       logits = pre_trained_model(**inputs).logits
#   return torch.softmax(logits, dim=1).numpy()

# # Choose a specific instance to explain
# idx = 0  # Index of the sample in your dataset
# text_instance = test_transformed_df_llm.iloc[idx]['text']

# # Create a LIME explainer
# explainer = LimeTextExplainer(class_names=["LEGITIMATE", "SUSPICIOUS"])

# # Generate explanation
# exp = explainer.explain_instance(text_instance, predictor)
# exp.show_in_notebook(text=True)

# COMMAND ----------

# # Ensure the model is in evaluation mode and moved to CPU
# model.eval()
# model.to('cpu')

# # Define a prediction function that only uses the CPU
# def predictor(texts):
#     # Adjust based on the structure of your dataset
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#     tokenizer.pad_token = '[PAD]'
#     inputs = tokenizer(texts['text'],return_tensors='pt', padding='max_length', truncation=True, max_length=128)
#     with torch.no_grad():
#         logits = model(**inputs).logits
#     return torch.softmax(logits, dim=1).numpy()  # No need to move to CPU as it's already ther

# COMMAND ----------

#Select Threshold Custom Function
def select_threshold(proba, target, fpr_max = 0.1 ):
    # calculate roc curves
    fpr, tpr, thresholds = roc_curve(target, proba)
    # get the best threshold with fpr <=0.1
    best_treshold = thresholds[fpr <= fpr_max][-1]
    
    return best_treshold

# COMMAND ----------

from sklearn.metrics import precision_recall_curve, roc_curve, precision_score, recall_score

# COMMAND ----------

target_fpr = 0.005
decision_threshold = select_threshold(y_prob, train_transformed_df_llm['label'].values, target_fpr)

# COMMAND ----------

print(decision_threshold)

# COMMAND ----------

import torch
model_predictions = trainer.predict(test_inputs)
logits = torch.tensor(model_predictions.predictions)
y_prob = torch.softmax(logits,dim=1).numpy()[:,1]

test_transformed_df_llm['prediction'] = (y_prob > decision_threshold).astype(int)

# COMMAND ----------

test_transformed_df_llm['prediction'].value_counts()

# COMMAND ----------

# test_transformed_df_llm['company_id'] = test_transformed_df_llm['company_id'].values
get_metrics(test_transformed_df_llm, test_transformed_df_llm['label'].values, test_transformed_df_llm['prediction'].values)

# COMMAND ----------

import os
from transformers.trainer_callback import TrainerState

ckpt_dirs = os.listdir(output_dir)
ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split('-')[1]))
last_ckpt = ckpt_dirs[-1]

state = TrainerState.load_from_json(f"{output_dir}/{last_ckpt}/trainer_state.json")

print(state.best_model_checkpoint) # your best ckpoint.

# COMMAND ----------

model_cpt = state.best_model_checkpoint
tokenizer_cpt = 'microsoft/MiniLM-L12-H384-uncased'
tuned_model = AutoModelForSequenceClassification.from_pretrained(f"{model_cpt}")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_cpt)
classify = pipeline(task='app-perp-ongoing-fraud_detection', model=tuned_model, tokenizer=tokenize_text)

# COMMAND ----------

all_files = glob.glob('inference_data/*')
for file_name in all_files:
    file = open(file_name)
    content = file.read()
    print(content)
    result = classify(content)
    print('PRED: ', result)
    print('GT: ', file_name.split('_')[-1].split('.txt')[0])
    print('\n')
