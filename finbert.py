# Set up Fine-Tuned FinBERT Model

!pip install transformers --quiet
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# load the FinBERT fine-tuned ESG 9-category model and the corresponding tokernizer (used for preprocessing input sentences)
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-esg-9-categories', num_labels = 9)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-esg-9-categories')

# load model on GPU/CPU
import torch
# load model on GPU if available
if torch.cuda.is_available():       
    device = torch.device("cuda")
    # put the model on GPU
    finbert.to(device)
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Model loaded on:', torch.cuda.get_device_name(0))

# load model on CPU if GPU is not available
else:
    device = torch.device("cpu")
    # put the model on CPU
    finbert.to(device)
    print('No GPU available, model loaded on CPU.')


# Load sample data set (225 sentences = 25 x 9 categories)

import pandas as pd

# import sample data
data = pd.read_csv('https://github.com/allenhhuang/FinBERT/raw/main/FinBERT%20test%20sample.csv').dropna(subset = ['Sentence', 'Label']).reset_index(drop = True)

# 9 categories and labels
label_map = {'Climate Change': 'CC',
             'Natural Capital': 'NC',
             'Pollution & Waste': 'PW',
             'Human Capital': 'HC',
             'Product Liability': 'PL',
             'Community Relations': 'CR',
             'Corporate Governance': 'CG',
             'Business Ethics & Values': 'BE',
             'Non-ESG': 'N'}

label_map_reverse = {'CC': 'Climate Change',
             'NC': 'Natural Capital',
             'PW': 'Pollution & Waste',
             'HC': 'Human Capital',
             'PL': 'Product Liability',
             'CR': 'Community Relations',
             'CG': 'Corporate Governance',
             'BE': 'Business Ethics & Values',
             'N': 'Non-ESG'}

# print the distribution of categories in sample data and first 5 observations
print(data['Label'].value_counts())
print('First 5 observations in sample data:')
data.head(5)


# Option 1: Output each category's probability

# define a function to output the probability of each category
from scipy.special import softmax
import numpy as np

def augmented_nlp(sentence):
  '''input any sentence, output FinBERT label, Predicted Label score/probability,
     and the probabilities that the sentence belongs to each category'''
  # preprocess the sentence
  input = tokenizer(sentence, return_tensors = 'pt', padding = True).to(device)
  # finbert raw output
  output = finbert(**input)[0]
  # convert the output to probabilities corresponding to each category
  probs = softmax(output.cpu().detach().numpy(), axis = 1)
  # get the numeric label with the highest probability
  value = np.argmax(probs, axis = -1)
  # get the textual label
  label = finbert.config.id2label[value[0]]
  # get the highest probability
  score = np.max(probs, axis = -1)

  return {'label': label,
          'score': score[0],
          'probabilities': probs[0].tolist()}

# demonstrate the output
print('augmented nlp output: ', augmented_nlp('For 2002, our total net emissions were approximately 60 million metric tons of CO2 equivalents for all businesses and operations we have ﬁnancial interests in, based on its equity share in those businesses and operations.'))


# Process sample data and download full output in a CSV file

# define the output structure
header = ['No.', 'Sentence', 'Label', 'FinBERT label', 'FinBERT score', 'Correct label'] + list(finbert.config.id2label.values())
result = []

# loop through the sentences
for i, row in data.iterrows():
  # sentence number
  id = str(i+1)
  print('Sentence number: ' + id)

  # sentence
  sentence = row['Sentence']
  print('Sentence: ' + sentence)

  # use the augmented_nlp function to process the sentence, get finbert label, score, and probabilities
  finbert_output = augmented_nlp(sentence)

  # label and score
  manual_label = row['Label']
  finbert_label = label_map[finbert_output['label']]
  finbert_score = finbert_output['score']
  print('Manual label: ' + label_map_reverse[manual_label])
  print('FinBERT label: ' + finbert_output['label'])
  print('FinBERT score: ', finbert_score)

  # probabilities
  finbert_probs = finbert_output['probabilities']

  print('-'*50)
  print()

  # combine all the output above into one row of observation
  row = [id, sentence, manual_label, finbert_label, finbert_score, 1*(manual_label==finbert_label)] + finbert_probs
  result.append(row)

# consolidate the output into a dataframe
result_df_full = pd.DataFrame(result, columns = header)

# save the output in the csv format
result_df_full.to_csv('FinBERT ESG9Class output full.csv', index = False)
# download the output to the local computer
from google.colab import files
files.download('FinBERT ESG9Class output full.csv')
# display the output
result_df_full




# Option 2: OPTION 2: Only Output Predicted Label
# use pipeline in transformers to assemble the steps of finbert prediction 
if torch.cuda.is_available(): # has GPU
  nlp = pipeline("text-classification", model = finbert, tokenizer = tokenizer, device = 0)
else: # CPU only
  nlp = pipeline("text-classification", model = finbert, tokenizer = tokenizer)


# demonstrate the output
print('nlp output: ', nlp('For 2002, our total net emissions were approximately 60 million metric tons of CO2 equivalents for all businesses and operations we have ﬁnancial interests in, based on its equity share in those businesses and operations.'))


# Process sample data and download predicted label

# define the output structure
header = ['No.', 'Sentence', 'Label', 'FinBERT label', 'FinBERT score', 'Correct label']
result = []

# loop through the sentences
for i, row in data.iterrows():

  # sentence number
  id = str(i+1)
  print('Sentence number: ' + id)

  # sentence
  sentence = row['Sentence']
  print('Sentence: ' + sentence)

  # use the nlp function to process the sentence, get finbert label and score
  finbert_output = nlp(sentence)[0]
  
  # label and score
  manual_label = row['Label']
  finbert_label = label_map[finbert_output['label']]
  finbert_score = finbert_output['score']
  print('Manual label: ' + label_map_reverse[manual_label])
  print('FinBERT label: ' + finbert_output['label'])
  print('FinBERT score: ', finbert_score)

  print('-'*50)
  print()

  # combine all the output above into one row of observation
  row = [id, sentence, manual_label, finbert_label, finbert_score, 1*(manual_label==finbert_label)]
  result.append(row)

# consolidate the output into a dataframe
result_df_label = pd.DataFrame(result, columns = header)

# save the output in the csv format
result_df_label.to_csv('FinBERT ESG9Class output label.csv', index = False)
# download the output to the local computer
from google.colab import files
files.download('FinBERT ESG9Class output label.csv')
# display the output
result_df_label



# Evaluate FinBERT performance (only if you already have label)
# evaluate the FinBERT performance
from sklearn.metrics import accuracy_score, precision_score, recall_score

# select which output to evaluate: result_df_full or result_df_label
# eval = result_df_full
eval = result_df_label

print('Accuracy score: ', accuracy_score(eval['Label'], eval['FinBERT label']))
print('Precision score: ', precision_score(eval['Label'], eval['FinBERT label'], average = 'macro'))
print('Recall score: ', recall_score(eval['Label'], eval['FinBERT label'], average = 'macro'))



# Try a single sentence
# paste any sentence here within the quotation marks
sentence = 'For 2002, our total net emissions were approximately 60 million metric tons of CO2 equivalents for all businesses and operations we have ﬁnancial interests in, based on its equity share in those businesses and operations.'
results = nlp(sentence)

# get FinBERT labels and scores
finbert_output = nlp(sentence)[0]
finbert_label = finbert_output['label']
finbert_score = finbert_output['score']

print('Sentence: ' + sentence)
print('FinBERT label: ' + finbert_label)
print('FinBERT score: ', finbert_score)



# Try an unlabeled dataset
!pip install transformers
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# load the FinBERT fine-tuned ESG 9-category model and the corresponding tokernizer
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-esg-9-categories', num_labels = 9)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-esg-9-categories')

# load model on GPU/CPU
import torch
# load model on GPU if available
if torch.cuda.is_available():       
    device = torch.device("cuda")
    # put the model on GPU
    finbert.to(device)
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Model loaded on:', torch.cuda.get_device_name(0))

# load model on CPU if GPU is not available
else:
    device = torch.device("cpu")
    # put the model on CPU
    finbert.to(device)
    print('No GPU available, model loaded on CPU.')
    
import pandas as pd

# import sample data
data = pd.read_csv('https://github.com/allenhhuang/FinBERT/raw/main/FinBERT%20test%20sample.csv').dropna(subset = ['Sentence']).reset_index(drop = True)

# 9 categories and labels
label_map = {'Climate Change': 'CC',
             'Natural Capital': 'NC',
             'Pollution & Waste': 'PW',
             'Human Capital': 'HC',
             'Product Liability': 'PL',
             'Community Relations': 'CR',
             'Corporate Governance': 'CG',
             'Business Ethics & Values': 'BE',
             'Non-ESG': 'N'}
             
# use pipeline in transformers to assemble the steps of finbert prediction 
if torch.cuda.is_available(): # has GPU
  nlp = pipeline("text-classification", model = finbert, tokenizer = tokenizer, device = 0)
else: # CPU only
  nlp = pipeline("text-classification", model = finbert, tokenizer = tokenizer)

# define the output structure
header = ['No.', 'Sentence', 'FinBERT label', 'FinBERT score']
result = []

# loop through the sentences
for i, row in data.iterrows():

  # sentence number
  id = str(i+1)
  # sentence
  sentence = row['Sentence']

  # use the nlp function to process the sentence, get finbert label and score
  finbert_output = nlp(sentence)[0]
  
  # label and score
  finbert_label = label_map[finbert_output['label']]
  finbert_score = finbert_output['score']
  
  # combine all the output above into one row of observation
  row = [id, sentence, finbert_label, finbert_score]
  result.append(row)

# consolidate the output into a dataframe
result_df_label = pd.DataFrame(result, columns = header)

# save the output in the csv format
result_df_label.to_csv('FinBERT ESG9Class prediction.csv', index = False)
# download the output to the local computer
from google.colab import files
files.download('FinBERT ESG9Class prediction.csv')
# display the output
result_df_label















from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# tested in transformers==4.18.0 
import transformers
print(transformers.__version__)


# Sentiment Analysis
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

nlp = pipeline("text-classification", model=finbert, tokenizer=tokenizer)
results = nlp(['growth is strong and we have plenty of liquidity.', 
               'there is a shortage of capital, and we need extra financing.',
              'formulation patents might protect Vasotec to a limited extent.'])

print(results)


# ESG-Classification
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-esg',num_labels=4)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-esg')

nlp = pipeline("text-classification", model=finbert, tokenizer=tokenizer)
results = nlp(['Managing and working to mitigate the impact our operations have on the environment is a core element of our business.',
               'Rhonda has been volunteering for several years for a variety of charitable community programs.',
               'Cabot\'s annual statements are audited annually by an independent registered public accounting firm.',
               'As of December 31, 2012, the 2011 Term Loan had a principal balance of $492.5 million.'])

print(results)


# FLS-Classification
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-fls',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-fls')

nlp = pipeline("text-classification", model=finbert, tokenizer=tokenizer)
results = nlp(['we expect the age of our fleet to enhance availability and reliability due to reduced downtime for repairs.',
               'on an equivalent unit of production basis, general and administrative expenses declined 24 percent from 1994 to $.67 per boe.',
               'we will continue to assess the need for a valuation allowance against deferred tax assets considering all available evidence obtained in future reporting periods.'])

print(results)
