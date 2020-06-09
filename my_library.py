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
    s_average = [[0]* 300]
  mv = meanv(s_average)
  return mv
