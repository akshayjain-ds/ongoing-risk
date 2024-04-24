# Databricks notebook source
# MAGIC %run ./utils_llm

# COMMAND ----------

train_transformed_df = load_dataset('/dbfs/Users/sri.duddu@tide.co/train_transformed_df_llm').toPandas()
test_transformed_df = load_dataset('/dbfs/Users/sri.duddu@tide.co/test_transformed_df_llm').toPandas()

# COMMAND ----------

train_transformed_df.shape, test_transformed_df.shape

# COMMAND ----------

TARGET_COLUMN = 'is_app_victim_counter_party'

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

# MAGIC %md
# MAGIC LLMs for Tabular Datasets

# COMMAND ----------

# MAGIC %md
# MAGIC Top Victim Features - 
# MAGIC
# MAGIC - registered_company_industry_classification
# MAGIC - number_of_tide_accounts_with_the_same_payee
# MAGIC - rolling_max_of_withdrawals_to_counter_party_60d
# MAGIC - number_of_withdrawals_over_threshold_60d
# MAGIC - rolling_sum_of_withdrawals_to_counter_party_60d
# MAGIC - ratio_pmts_outside_business_60d
# MAGIC - rolling_sum_of_withdrawals_value_to_counter_party_abv_thresh_1000_60d
# MAGIC - payment_transfer_rate_on_cop
# MAGIC - rolling_max_of_deposits_from_counter_party_60d
# MAGIC - number_of_high_outgoing_fastpayments_on_account_60d
# MAGIC - ratio_of_pmts_value_to_payee_of_overall_pmts_60d
# MAGIC - rolling_sum_of_withdrawals_on_account_2h
# MAGIC - rolling_mean_of_deposits_from_counter_party_60d

# COMMAND ----------

import math

def pre_concatenate_text(x):
    if math.isnan(x['company_core_features_v3__is_registered']):
      ic = f"This member account is not registered, "
    else:
      ic = f"This member account is registered with {x['ifre_member_features_v2__industry_classification']} industry classification, "
    
    if math.isnan(x['payment_recipient_events_features_new_to_tide__number_of_tide_accounts_with_same_payee']):
      ntasp = f"has no tide accounts with the same payee, "
    else:
      ntasp = f"has {int(x['payment_recipient_events_features_new_to_tide__number_of_tide_accounts_with_same_payee'])} tide accounts with the same payee, "

    full_text = (
        f"{ic}",
        f"{ntasp}"
    )
    return ''.join(full_text)

def post_concatenate_text(x):
    full_text = (
        f"had done a transaction of {int(x['requested_payment_value'])} value in the first {int(x['hours_since_confirmation_of_payee'])} hours after payee addition. " ,
        f"The maximum of withdrawals done earlier to the payee in the last 60 days is {int(x['rolling_max_of_withdrawals_to_counter_party_60d'])}. ",
        f"Also, the overall number of withdrawals over a high threshold in the last 60 days is {int(x['number_of_withdrawals_over_threshold_60d'])}. "
        f"Ratio of payments outside of the business in the last 60 days is {int(x['ratio_pmts_outside_business_60d'])}. "
        f"The average of deposits done earlier from the payee in the last 60 days is {int(x['rolling_mean_of_deposits_from_counter_party_60d'])}. "
        f"The number of high outgoing fast payments on the member account in the last 60 days is {int(x['number_of_high_outgoing_fastpayments_on_account_60d'])} "
        f"and total sum of withdrawals on the member account in the last 2 hours is {int(x['rolling_sum_of_withdrawals_on_account_2h'])}"
    )
    return ''.join(full_text)


train_transformed_df['label'] = train_transformed_df[TARGET_COLUMN].values
test_transformed_df['label'] = test_transformed_df[TARGET_COLUMN].values

train_transformed_df['pre_trans_text'] = train_transformed_df.apply(lambda x: pre_concatenate_text(x), axis=1).values
test_transformed_df['pre_trans_text'] = test_transformed_df.apply(lambda x: pre_concatenate_text(x), axis=1).values
train_transformed_df['post_trans_text'] = train_transformed_df.apply(lambda x: post_concatenate_text(x), axis=1).values
test_transformed_df['post_trans_text'] = test_transformed_df.apply(lambda x: post_concatenate_text(x), axis=1).values

train_transformed_df['text'] =  train_transformed_df['pre_trans_text'] + train_transformed_df['post_trans_text']
test_transformed_df['text'] =  test_transformed_df['pre_trans_text'] + test_transformed_df['post_trans_text']

# COMMAND ----------

train_transformed_df['text'].iloc[0]

# COMMAND ----------

test_transformed_df['text'].iloc[0]

# COMMAND ----------

from transformers import Trainer
import torch

class MyTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # You pass the class weights when instantiating the Trainer
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.

            # Changes start here
            # loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            logits = outputs['logits']
            criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = criterion(logits, inputs['labels'])
            # Changes end here

        return (loss, outputs) if return_outputs else loss

# COMMAND ----------

train_transformed_df_llm = train_transformed_df.copy()
test_transformed_df_llm = test_transformed_df.copy()

# COMMAND ----------

train_transformed_df_llm.shape, test_transformed_df_llm.shape

# COMMAND ----------

train_transformed_df_llm['label']

# COMMAND ----------

gc.collect()
del train_transformed_df, test_transformed_df

# COMMAND ----------

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import datasets
import torch as t
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np

from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PromptEncoderConfig,
)

def load_LLM(llm, device):
    num_labels = 2
    # Define label mappings
    id2label = {0: "LEGITIMATE", 1: "SUSPICIOUS"}
    label2id = {"LEGITIMATE": 0, "SUSPICIOUS": 1}
    model = AutoModelForSequenceClassification.from_pretrained(llm,num_labels=num_labels,id2label=id2label, label2id=label2id)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(llm)
    return model, tokenizer

# llm = "EleutherAI/gpt-neo-2.7B"
llm = "microsoft/MiniLM-L12-H384-uncased"
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
model,tokenizer = load_LLM(llm,device)

# peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=20, encoder_hidden_size=128)

# model = get_peft_model(model, peft_config)

def tokenize_function(examples):
    # Adjust based on the structure of your dataset
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = '[PAD]'
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

# COMMAND ----------

from datasets import Dataset

# # Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_transformed_df_llm[['text', 'label']])
test_dataset = Dataset.from_pandas(test_transformed_df_llm[['text', 'label']])

train_inputs = train_dataset.map(tokenize_function, batched=True)
test_inputs = test_dataset.map(tokenize_function, batched=True)

# Format the datasets correctly with labels
train_inputs = train_inputs.map(lambda x: {'labels': x['label']})
test_inputs = test_inputs.map(lambda x: {'labels': x['label']})

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# COMMAND ----------

test_inputs

# COMMAND ----------

import torch
torch.cuda.empty_cache() 

# COMMAND ----------

f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="my_awesome_model_test",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    weight_decay=0.01,
    evaluation_strategy="steps",
    load_best_model_at_end = True,
    greater_is_better=True
)

# model.config.pad_token_id = tokenizer.pad_token_id

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_inputs,
#     eval_dataset=test_inputs,
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics
# )

# trainer.train()

# Define the trainer
trainer = MyTrainer(
    model=model,
    class_weights = torch.tensor([1.0, 100.0], device=model.device),
    args=training_args,
    train_dataset=train_inputs,
    eval_dataset=test_inputs,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# COMMAND ----------

# Evaluate the model
results = trainer.evaluate()
print(results)

# COMMAND ----------

model_predictions = trainer.predict(train_inputs)

# COMMAND ----------

import torch

logits = torch.tensor(model_predictions.predictions)
y_prob = torch.softmax(logits,dim=1).numpy()[:,1]

# COMMAND ----------

y_prob

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