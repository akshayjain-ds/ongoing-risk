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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
gc.enable()


# COMMAND ----------

features = pd.read_csv('/dbfs/dbfs/Shared/Decisioning/Strawberries/App_fraud/ntt_ftt_random_day_raw_features_2022-01-01_2023-12-31.csv.gz',
            dtype={id1: str}, memory_map=True)
features = features.assign(
  timestamp = lambda col: pd.to_datetime(col[timestamp])
  )

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
  .apply(lambda x: x.replace('category.', '')),
  .apply(lambda x: re.sub('[^a-zA-Z0-9 \n\.]', ' ', x))
  .apply(lambda x: _RE_COMBINE_WHITESPACE.sub(' ', x))
  .apply(lambda x: _RE_STRIP_WHITESPACE.sub('', x)),

  applicant_id_type = lambda col: col['applicant_id_type']
  .apply(lambda x: re.sub('[^a-zA-Z0-9 \n\.]', ' ', str(x))),

  company_type = lambda col: col['company_type']
  .apply(lambda x: re.sub('[^a-zA-Z0-9 \n\.]', ' ', x)),

  applicant_idcountry_issue = lambda col: col['applicant_idcountry_issue'].fillna('GB')
  .apply(lambda x: pycountry.countries.get(alpha_2=x).name),

  fastpmt_beneficiaries = lambda col: col['fastpmt_beneficiaries'].fillna(0),
  fastpmt_benefactors = lambda col: col['fastpmt_benefactors'].fillna(0),
  ddebit_beneficiaries = lambda col: col['ddebit_beneficiaries'].fillna(0),
)      
features.shape

# COMMAND ----------

features = pd.concat([features[features.is_app_fraud_45d==1], 
                      features[features.is_app_fraud_45d==0].sample(frac=0.05, random_state=seed)]).sample(frac=1.0, random_state=seed)
features.shape

# COMMAND ----------

import math

def concatenate_text(x):
  
  if math.isnan(x['company_is_registered']):
      ic = f"This member account is of type {x['company_type']} and is not registered. "
  else:
    ic = f"This member account is of type {x['company_type']} and is registered with {x['company_icc']} as industry classification. "
  
  if math.isnan(x['company_age_at_timestamp']):
    age = f"The age of applicant at onboarding is {int(x['applicant_age_at_completion'])} years and age of business is unknown. "
  else:
    age = f"The age of applicant at onboarding is {int(x['applicant_age_at_completion'])} years and age of business is {int(x['company_age_at_timestamp'])} months. "

  if x['is_restricted_keyword_present']==1:
    c_name = f"Business has high risk of fraudulent activity due to presence of restricted keywords in its trading name. "
  else:
    c_name = f"Business has low or medium risk of fraudulent activity due to absense of restricted keywords in its trading name. "

  if math.isnan(x['days_to_transact']):
    ntt_ftt = f"Member has not done first transaction till now since approval, "
  else:
    ntt_ftt = f"Member has done first transaction after {int(x['days_to_transact'])} days since approval, "

  if math.isnan(x['applicant_years_to_id_expiry']):
    expiry = f"valid for unknown years for its expiry. "
  else:
    expiry = f"valid for {int(x['applicant_years_to_id_expiry'])} years. "

  full_text = (
    f"{ic}",
    f"{age}",
    f"{c_name}",
    f"{ntt_ftt}",
    f"has taken {int(x['days_to_approval'])} days for getting approved after completeing onboarding event. ",
    f"Member has completed {int(x['days_on_books'])} days on books since approval ",
    f"and {int(x['days_remaining_as_ntt_ftt'])} days are remaining for period in consideration. ",
    f"Applicant email contains {'no' if ~x['applicant_email_numeric'] else 'some'} numeric characters with domain {x['applicant_email_domain']} and used {x['applicant_device_type']} operating system on mobile device. ",
    f"The proof of identification submitted is {x['applicant_id_type']} issued by {x['applicant_idcountry_issue']} country and {expiry}",
    f"The number of fast payment beneficiaries added is {int(x['fastpmt_beneficiaries'])}, number of fast payment benefactors added is {int(x['fastpmt_benefactors'])} and number of direct debit beneficiaries added is {int(x['ddebit_beneficiaries'])}. "
  )
  return ''.join(full_text).strip()

# COMMAND ----------

features['label'] = features['is_app_fraud_45d'].astype(int)
features['text'] = features.progress_apply(lambda row: concatenate_text(row), axis=1).values

# COMMAND ----------

features['text'].iloc[0]

# COMMAND ----------

max_length = len(features['text'].iloc[0].split())
max_length = 148

# COMMAND ----------

train_index = features[timestamp] <= pd.to_datetime(train_end_date) 
test_index = features[timestamp] > pd.to_datetime(test_start_date) 
train_transformed_df_llm = features[train_index][['text', 'label']]
test_transformed_df_llm = features[test_index][['text', 'label']]
train_transformed_df_llm['label'] = train_transformed_df_llm['label']
test_transformed_df_llm['label'] = test_transformed_df_llm['label']
train_transformed_df_llm.shape, test_transformed_df_llm.shape

# COMMAND ----------

train_transformed_df_llm.info()

# COMMAND ----------

train_transformed_df_llm.shape, test_transformed_df_llm.shape

# COMMAND ----------

train_transformed_df_llm['label'].mean(), test_transformed_df_llm['label'].mean()

# COMMAND ----------


class_weights = sklearn.utils.class_weight.compute_class_weight('balanced', [0,1], train_transformed_df_llm['label']).tolist()
class_weights

# COMMAND ----------

import torch
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
#Additional Info when using cuda
if device.type == 'cuda':
  print(torch.cuda.get_device_name(0))
  print('Memory Usage:')
  print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
  print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
torch.cuda.empty_cache()
gc.collect()

# COMMAND ----------


# Preparing the Text for Our Model
model_cpt = 'microsoft/MiniLM-L12-H384-uncased'
pre_trained_tokenizer = AutoTokenizer.from_pretrained(model_cpt)

def tokenize_text(examples, tokenizer=pre_trained_tokenizer):

    return tokenizer(examples['text'], truncation=True, max_length=150, padding='max_length')

pre_trained_model = AutoModelForSequenceClassification.from_pretrained(model_cpt, 
                                                          num_labels=2, 
                                                          id2label = {0: "LEGITIMATE", 1: "SUSPICIOUS"},
                                                          label2id = {"LEGITIMATE": 0, "SUSPICIOUS": 1})
pre_trained_model.to(device)

# COMMAND ----------


# # Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_transformed_df_llm)
test_dataset = Dataset.from_pandas(test_transformed_df_llm)

train_inputs = train_dataset.map(tokenize_text, batched=True)
test_inputs = test_dataset.map(tokenize_text, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=pre_trained_tokenizer)

# COMMAND ----------

train_inputs

# COMMAND ----------

test_inputs

# COMMAND ----------

batch_size = 64
logging_steps = len(train_inputs) // batch_size
output_dir = './ntt_ftt_ongoing_app_perp'
epochs=10

# COMMAND ----------

training_args = TrainingArguments(
    disable_tqdm=False,
    output_dir=output_dir,
    lr_scheduler_type = 'cosine_with_restarts',
    learning_rate=1e-4,
    warmup_ratio=0.05,
    weight_decay=0.1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    evaluation_strategy="steps",
    load_best_model_at_end = True,
    greater_is_better=True,
    logging_dir='./logs',
)
training_args.device

# COMMAND ----------

def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  
  # Calculate accuracy
  accuracy = accuracy_score(labels, preds)

  # Calculate precision, recall, and F1-score
  precision = precision_score(labels, preds, average='weighted')
  recall = recall_score(labels, preds, average='weighted')
  f1 = f1_score(labels, preds, average='weighted')
  
  return {
      'accuracy': accuracy,
      'precision': precision,
      'recall': recall,
      'f1': f1
  }

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
        
        labels = inputs.get("labels")
        
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        # compute custom loss
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights, reduction='mean')
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

trainer = CWTrainer(
    class_weights = torch.tensor(class_weights, device=device),
    model=pre_trained_model,
    args=training_args,
    train_dataset=train_inputs,
    eval_dataset=test_inputs,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
)

trainer.train()

# COMMAND ----------

# Evaluate the model
results = trainer.evaluate()
print(results)

# COMMAND ----------

y_pred_train = trainer.predict(train_inputs)
y_pred_test = trainer.predict(test_inputs)

# COMMAND ----------

y_pred_train

# COMMAND ----------

logits = torch.tensor(y_pred_train.predictions)
y_pred_train = torch.sigmoid(logits, dim=1).numpy()[:, 1]
logits = torch.tensor(y_pred_test.predictions)
y_pred_test = torch.sigmoid(logits, dim=1).numpy()[:, 1]

# COMMAND ----------

y_pred_train.mean(), y_pred_test.mean()

# COMMAND ----------

from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(train_transformed_df_llm.label, y_pred_train)
print(auc(fpr, tpr))
fpr, tpr, thresholds = roc_curve(test_transformed_df_llm.label, y_pred_test)
print(auc(fpr, tpr))

# COMMAND ----------

# from lime.lime_text import LimeTextExplainer
# import torch

# # Ensure the model is in evaluation mode and moved to CPU
# model.eval()
# model.to('cpu')

# # Define a prediction function that only uses the CPU
# def predictor(texts):
#     inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
#     with torch.no_grad():
#         logits = model(**inputs).logits
#     return torch.softmax(logits, dim=1).numpy()  # No need to move to CPU as it's already there

# # Create a LIME explainer
# explainer = LimeTextExplainer(class_names=["LEGITIMATE", "SUSPICIOUS"])

# # Choose a specific instance to explain
# idx = 0  # Index of the sample in your dataset
# text_instance = test_transformed_df_llm.iloc[idx]['text']

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