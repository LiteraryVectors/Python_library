import pandas as pd
import numpy
from typing import TypeVar, Callable
narray = TypeVar('numpy.ndarray')
import numpy as np
import spacy
spnlp = TypeVar('spacy.lang.en.English')  #for type hints
import os
os.system("python -m spacy download en_core_web_md")
import en_core_web_md
nlp = en_core_web_md.load()

def hello():
  print('hello')
def foo():
  print('foo')
  
  #Vector Functions: 
    
def dividev(x:list, c) -> list:
  assert isinstance(x, list), f"x must be a list but instead is {type(x)}"
  assert isinstance(c, int) or isinstance(c, float), f"c must be an int or a float but instead is {type(c)}"

  #result = [v/c for v in x]  #one-line compact version

  result = []
  for i in range(len(x)):
    v = x[i]
    result.append(v/c) #division produces a float

  return result

def addv(x:list, y:list) -> list:
  assert isinstance(x, list), f"x must be a list but instead is {type(x)}"
  assert isinstance(y, list), f"y must be a list but instead is {type(y)}"
  assert len(x) == len(y), f"x and y must be the same length"

  #result = [c1 + c2 for c1, c2 in zip(x, y)]  #one-line compact version

  result = []
  for i in range(len(x)):
    c1 = x[i]
    c2 = y[i]
    result.append(c1+c2)

  return result

def meanv(matrix: list) -> list:
    assert isinstance(matrix, list), f"matrix must be a list but instead is {type(x)}"
    assert len(matrix) >= 1, f'matrix must have at least one row'

    #Python transpose: sumv = [sum(col) for col in zip(*matrix)]

    sumv = matrix[0]  #use first row as starting point in "reduction" style
    for row in matrix[1:]:   #make sure start at row index 1 and not 0
      sumv = addv(sumv, row)
    mean = dividev(sumv, len(matrix))
    return mean
#Word Embeddings Fuctions:

def ordered_embeddings(target_vector, table):
  names = table.index.tolist()
  ordered_list = []
  for i in range(len(names)):
    name = names[i]
    row = table.loc[name].tolist()
    d = up.euclidean_distance(target_vector, row)
    ordered_list.append([d, names[i]])
  ordered_list = sorted(ordered_list)

  return ordered_list

def get_vec(s:str) -> list:
    return nlp.vocab[s].vector.tolist()
  
def sent2vec (s:str) -> list:
  doc = nlp(s)
  s_average = []
  for token in doc:
    if token.is_alpha and not token.is_stop:
      v = get_vec(token.text)
      s_average.append(v)
  if len(s_average)== 0:
    
  #Other
  
  def bayes_gothic_tester(testing_table:dframe, evidence_bag:dframe, training_table:dframe, laplace:float=1.0) -> list:
      assert isinstance(testing_table, pd.core.frame.DataFrame), f'test_table not a dataframe but instead a {type(testing_table)}'
      assert isinstance(evidence_bag, pd.core.frame.DataFrame), f'evidence_bag not a dframe but instead a {type(evidence_bag)}'
      assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table not a dataframe but instead a {type(training_table)}'
      assert 'author' in training_table, f'author column is not found in training_table'
      assert 'text' in testing_table, f'text column is not found in testing_table'

  result_list = []
  for i,target_row in testing_table.iterrows():
    raw_text = target_row['text']  #a sentence
    doc = nlp(raw_text.lower())  #create the tokens

    evidence_list = []
    for token in doc:
      if not token.is_alpha or token.is_stop: continue
      evidence_list.append(token.text)

    p_tuple = bayes_laplace(list(set(evidence_list)), evidence_bag, training_table, laplace)
    result_list.append(p_tuple)
  return result_list

def bayes(evidence:set, evidence_bag:dict, training_table:dframe) -> tuple:
  assert isinstance(evidence, set), f'evidence not a set but instead a {type(evidence)}'
  assert isinstance(evidence_bag, dict), f'evidence_bag not a dict but instead a {type(evidence_bag)}'
  assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table not a dataframe but instead a {type(training_table)}'
  assert 'label' in training_table, f'label column is not found in training_table'
  assert training_table.label.dtype == int, f"label must be an int column (possibly wrangled); instead it has type({training_table.label.dtype})"

  label_list = training_table.label.to_list()
  n_classes = len(set(label_list))
  assert len(list(evidence_bag.values())[0]) == n_classes, f'Values in evidence_bag do not match number of unique classes ({n_classes}) in labels.'

  counts = []
  probs = []
  for i in range(n_classes):
    ct = label_list.count(i)
    counts.append(ct)
    probs.append(ct/len(label_list))

  #now have counts and probs for all classes
  #CONSIDER CHANGING TO LN OF PRODUCTS. END UP SUMMING LOGS OF EACH ITEM. AVOIDS UNDERFLOW.

  results = []
  for a_class in range(n_classes):
    numerator = 1
    for ei in evidence:
      all_values = evidence_bag[ei]
      the_value = (all_values[a_class]/counts[a_class])
      numerator *= the_value
    results.append(numerator * probs[a_class])

  return tuple(results)

def char_set_builder(text:str) -> list:
  the28 = set(text).intersection(set('abcdefghijklmnopqrstuvwxyz!#'))
  return list(the28)

def bayes_tester(testing_table:dframe, evidence_bag:dict, training_table:dframe, parser:Callable) -> list:
  assert isinstance(testing_table, pd.core.frame.DataFrame), f'test_table not a dataframe but instead a {type(testing_table)}'
  assert isinstance(evidence_bag, dict), f'evidence_bag not a dict but instead a {type(evidence_bag)}'
  assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table not a dataframe but instead a {type(training_table)}'
  assert callable(parser), f'parser not a function but instead a {type(parser)}'
  assert 'label' in training_table, f'label column is not found in training_table'
  assert training_table.label.dtype == int, f"label must be an int column (possibly wrangled); instead it has type({training_table.label.dtype})"
  assert 'text' in testing_table, f'text column is not found in testing_table'


  result_list = []
  for i,target_row in testing_table.iterrows():
    raw_text = target_row['text']
    e_set = set(parser(raw_text))
    p_tuple = bayes(e_set, evidence_bag, training_table)
    result_list.append(p_tuple)
  return result_list

'''
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

swords = stopwords.words('english')
swords.sort()

import re
def get_clean_words(stopwords:list, raw_sentence:str) -> list:
  assert isinstance(stopwords, list), f'stopwords must be a list but saw a {type(stopwords)}'
  assert all([isinstance(word, str) for word in stopwords]), f'expecting stopwords to be a list of strings'
  assert isinstance(raw_sentence, str), f'raw_sentence must be a list but saw a {type(raw_sentence)}'

  sentence = raw_sentence.lower()
  for word in stopwords:
    sentence = re.sub(r"\b"+word+r"\b", '', sentence)  #replace stopword with empty

  cleaned = re.findall("\w+", sentence)  #now find the words
  return cleaned

def build_word_bag(stopwords:list, training_table:dframe) -> dict:
  assert isinstance(stopwords, list), f'stopwords must be a list but saw a {type(stopwords)}'
  assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table not a dataframe but instead a {type(training_table)}'

  bow = {}
  starters = [[1,0,0], [0,1,0], [0,0,1]]
  for i,row in training_table.iterrows():
    raw_text = row['text']
    words = set(get_clean_words(stopwords, raw_text))
    label =  row['label']
    for word in words:
        if word in bow:
            bow[word][label] += 1
        else:
            bow[word] = list(starters[label])  #need list to get a copy
  return bow
'''

def robust_bayes(evidence:set, evidence_bag:dict, training_table:dframe, laplace:float=1.0) -> tuple:
  assert isinstance(evidence, set), f'evidence not a set but instead a {type(evidence)}'
  assert isinstance(evidence_bag, dict), f'evidence_bag not a dict but instead a {type(evidence_bag)}'
  assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table not a dataframe but instead a {type(training_table)}'
  assert 'label' in training_table, f'label column is not found in training_table'
  assert training_table.label.dtype == int, f"label must be an int column (possibly wrangled); instead it has type({training_table.label.dtype})"

  label_list = training_table.label.to_list()
  n_classes = len(set(label_list))
  assert len(list(evidence_bag.values())[0]) == n_classes, f'Values in evidence_bag do not match number of unique classes ({n_classes}) in labels.'

  counts = []
  probs = []
  for i in range(n_classes):
    ct = label_list.count(i)
    counts.append(ct)
    probs.append(ct/len(label_list))

  #now have counts and probs for all classes

  results = []
  for a_class in range(n_classes):
    numerator = 1
    for ei in evidence:
      if ei not in evidence_bag:
        the_value =  1/(counts[a_class] + len(evidence_bag) + laplace)
      else:
        all_values = evidence_bag[ei]
        the_value = ((all_values[a_class]+laplace)/(counts[a_class] + len(evidence_bag) + laplace)) 
      numerator *= the_value
    results.append(max(numerator * probs[a_class], 2.2250738585072014e-308))

  return tuple(results)

def robust_bayes_tester(testing_table:dframe, evidence_bag:dict, training_table:dframe, parser:Callable) -> list:
  assert isinstance(testing_table, pd.core.frame.DataFrame), f'test_table not a dataframe but instead a {type(testing_table)}'
  assert isinstance(evidence_bag, dict), f'evidence_bag not a dict but instead a {type(evidence_bag)}'
  assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table not a dataframe but instead a {type(training_table)}'
  assert callable(parser), f'parser not a function but instead a {type(parser)}'
  assert 'label' in training_table, f'label column is not found in training_table'
  assert training_table.label.dtype == int, f"label must be an int column (possibly wrangled); instead it has type({training_table.label.dtype})"
  assert 'text' in testing_table, f'text column is not found in testing_table'

  result_list = []
  for i,target_row in testing_table.iterrows():
    raw_text = target_row['text']
    e_set = set(parser(raw_text))
    p_tuple = robust_bayes(e_set, evidence_bag, training_table)
    result_list.append(p_tuple)
  return result_list

