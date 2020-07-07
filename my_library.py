import pandas as pd
import numpy
from numpy.linalg import norm
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

def fast_cosine(v1:narray, v2:narray) -> float:
  assert isinstance(v1, numpy.ndarray), f"v1 must be a numpy array but instead is {type(v1)}"
  assert len(v1.shape) == 1, f"v1 must be a 1d array but instead is {len(v1.shape)}d"
  assert isinstance(v2, numpy.ndarray), f"v2 must be a numpy array but instead is {type(v2)}"
  assert len(v2.shape) == 1, f"v2 must be a 1d array but instead is {len(v2.shape)}d"
  assert len(v1) == len(v2), f'v1 and v2 must have same length but instead have {len(v1)} and {len(v2)}'

  x = norm(v1)
  if x==0: return 0.0
  y = norm(v2)
  if y==0: return 0.0
  z = x*y
  if z==0: return 0.0  #check for underflow
  return np.dot(v1, v2)/z

def cosine_similarity(vect1:list ,vect2:list) -> float:
  assert isinstance(vect1, list), f'vect1 is not a list but a {type(vect1)}'
  assert isinstance(vect2, list), f'vect2 is not a list but a {type(vect2)}'
  assert len(vect1) == len(vect2), f"Mismatching length for vectors: {len(vect1)} and {len(vect2)}"
  '''
  sumxx, sumxy, sumyy = 0, 0, 0
  for i in range(len(vect1)):
      x = vect1[i]; y = vect2[i]
      sumxx += x*x
      sumyy += y*y
      sumxy += x*y
      denom = sumxx**.5 * sumyy**.5  #or (sumxx * sumyy)**.5
  #have to invert to order on smallest

  return sumxy/denom if denom > 0 else 0.0
  '''
  return fast_cosine(np.array(vect1), np.array(vect2)).tolist()

def inverse_cosine_similarity(vect1:list ,vect2:list) -> float:
  assert isinstance(vect1, list), f'vect1 is not a list but a {type(vect1)}'
  assert isinstance(vect2, list), f'vect2 is not a list but a {type(vect2)}'
  assert len(vect1) == len(vect2), f"Mismatching length for vectors: {len(vect1)} and {len(vect2)}"

  normal_result = cosine_similarity(vect1, vect2)
  return 1.0 - normal_result

def build_word_table(books:dict):
  assert isinstance(books, dict), f'books not a dictionary but instead a {type(books)}'

  all_titles = list(books.keys())
  n = len(all_titles)
  word_table = pd.DataFrame(columns=['word'] + all_titles)
  m = max([len(v)  for v in books.values()])  #Number of characters in longest book
  nlp.max_length = m

  for i,title in enumerate(all_titles):
    print(f'({i+1} of {n}) Processing {title} ({len(books[title])} characters)')
    doc = nlp(books[title].lower()) #parse the entire book into tokens
    out = display(progress(0, len(doc)), display_id=True)
    cut = int(len(doc)*.1)
    for j,token in enumerate(doc):
      if  token.is_alpha and not token.is_stop:
        word_table = update_word_table(word_table, token.text, title)
      if j%cut==0:
        out.update(progress(j+1, len(doc)))  #shows progress bar
        time.sleep(0.02)

  word_table = word_table.infer_objects()
  #word_table = word_table.astype(int)  #all columns
  word_table = word_table.astype({'word':str})  #now just word column

  sorted_word_table = word_table.sort_values(by=['word'])
  sorted_word_table = sorted_word_table.reset_index(drop=True)
  sorted_word_table = sorted_word_table.set_index('word')  #set the word column to be the table index

  return sorted_word_table

def most_similar_word(word_table, target_word:str) -> list:
  assert isinstance(word_table, pd.core.frame.DataFrame), f'word_table not a dframe but instead a {type(word_table)}'

  target_vec = list(nlp.vocab.get_vector(target_word))
  distance_list = []
  word_list = word_table.index.to_list()
  out = display(progress(0, len(word_list)), display_id=True)
  cut = int(len(word_list)*.1)
  for i,word in enumerate(word_list):
    vec = list(nlp.vocab.get_vector(word))
    d = euclidean_distance(target_vec, vec)
    distance_list.append([word, d])
    if i%cut==0:
      out.update(progress(i+1, len(word_list)))  #shows progress bar
      time.sleep(0.02)
  ordered = sorted(distance_list, key=lambda p: p[1])
  return ordered

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
#%tensorflow_version 2.x
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras import Sequential
from tensorflow.keras.utils import plot_model

from sklearn.model_selection import GridSearchCV
#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

#libraries to help visualize training results later
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
#%matplotlib inline
rcParams['figure.figsize'] = 10,8

#Used to show progress bar in loop
from IPython.display import HTML, display
import time
def progress(value, max=100):
    return HTML("""
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(value=value, max=max))

def ann_build_model(n:int, layer_list: list, seed=1234, metrics='binary_accuracy'):
  assert isinstance(n, int), f'n is an int, the number of columns/features of each sample. Instead got {type(n)}'
  assert isinstance(layer_list, list) or isinstance(layer_list, tuple), f'layer_list is a list or tuple, the number of nodes per layer. Instead got {type(layer_list)}'

  if len(layer_list) == 1:
    print('Warning: layer_list has only 1 layer, the output layer. So no hidden layers')

  if layer_list[-1] != 1:
    print(f'Warning: layer_list has more than one node in the output layer: {layer_list[-1]}')

  np.random.seed(seed=seed)
  tf.random.set_seed(seed)

  model = Sequential()  #we will always use this in our class. It means left-to-right as we have diagrammed.
  model.add(Dense(units=layer_list[0], activation='sigmoid', input_dim=n))  #first hidden layer needs number of inputs
  for u in layer_list[1:]:
    model.add(Dense(units=u, activation='sigmoid'))

  loss_choice = 'binary_crossentropy'
  optimizer_choice = 'sgd'
  model.compile(loss=loss_choice,
              optimizer=optimizer_choice,
              metrics=[metrics])  #metrics is just to help us to see what is going on. kind of debugging info.
  return model

def ann_train(model, x_train:list, y_train:list, epochs:int,  batch_size=1):
  assert isinstance(x_train, list), f'x_train is a list, the list of samples. Instead got {type(x_train)}'
  assert isinstance(y_train, list), f'y_train is a list, the list of samples. Instead got {type(y_train)}'
  assert len(x_train) == len(y_train), f'x_train must be the same length as y_train'
  assert isinstance(epochs, int), f'epochs is an int, the number of epochs to repeat. Instead got {type(epochs)}'
  assert model.get_input_shape_at(0)[1] == len(x_train[0]), f'model expecting sample size of {model.get_input_shape_at(0)[1]} but saw {len(x_train[0])}'
  
  if epochs == 1:
    print('Warning: epochs is 1, typically too small.')

  xnp = np.array(x_train)
  ynp = np.array(y_train)
  training = model.fit(xnp, ynp, epochs=epochs, batch_size=batch_size, verbose=0)  #3 minutes
  
  plt.plot(training.history['binary_accuracy'])
  plt.plot(training.history['loss'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['binary accuracy', 'loss'], loc='upper left')
  plt.show()
  return training

#for grid search
def create_model(input_dim=300, lyrs=(64,32)):
    model = ann_build_model(input_dim, lyrs, metrics='accuracy')
    return model
  
def grid_search(layers_list, epochs_list, X_train, Y_train, indim=300):
  tup_layers = tuple([tuple(l) for l in layers_list])
  tup_epochs = tuple(epochs_list)
  
  model = KerasClassifier(build_fn=create_model, verbose=0)  #use our create_model
  
  # define the grid search parameters
  batch_size = [1]  #starting with just a few choices
  epochs = tup_epochs
  lyrs = tup_layers

  #use this to override our defaults. keys must match create_model args
  param_grid = dict(batch_size=batch_size, epochs=epochs, input_dim=[indim], lyrs=lyrs)

  # buld the search grid
  grid = GridSearchCV(estimator=model,   #we created model above
                      param_grid=param_grid,
                      cv=3,  #use 3 folds for cross-validation
                      verbose=2)  # include n_jobs=-1 if you are using CPU
  
  grid_result = grid.fit(np.array(X_train), np.array(Y_train))
  
  # summarize results
  print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
  means = grid_result.cv_results_['mean_test_score']
  stds = grid_result.cv_results_['std_test_score']
  params = grid_result.cv_results_['params']
  for mean, stdev, param in zip(means, stds, params):
      print("%f (%f) with: %r" % (mean, stdev, param))
