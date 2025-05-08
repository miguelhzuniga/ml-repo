## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import itertools, numpy, os, sys
from sklearn.linear_model import LogisticRegression as Model
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import joblib
sys.path.append( os.path.dirname( os.path.abspath( __file__ ) ) )
from ModelBasedPlayer import ModelBasedPlayer
from utils import *

"""
"""
class Player( ModelBasedPlayer ):

  '''
  '''
  def __init__( self, args ):
    super( ).__init__( args )
    # print("inicializacion")
  # end def

  '''
  '''
  def read_model( self, fname ):
    # Read model
    model = joblib.load(fname)

    # Save or update the attribute m_Model
    self.m_Model = model
  # end def

  '''
  '''
  def evaluate( self, X , plays_len, n_mines):
    # print(calculated_mine_prob_matrix(X))

    contains_zero = (X == 0).any(axis=1)
    mine_rule_one = ((X >= 1) & (X != 9)).all(axis=1)
    mine_rule_two = (
      (numpy.sum(X == -1, axis=1) == 3)
      & (numpy.sum((X >= 1) & (X != 9), axis=1) == 5)
    )
    mine_rule_three = (
      (numpy.sum(X == -1, axis=1) == 5)
      & (numpy.sum((X >= 1) & (X != 9), axis=1) == 3)
    )
    contains_no_info = (numpy.sum(X == -1, axis=1) + numpy.sum(X == 9, axis=1)) == 8

    identified_mines = numpy.sum(mine_rule_one == True)
    identified_mines += numpy.sum(mine_rule_two == True)
    identified_mines += numpy.sum(mine_rule_three == True)

    if identified_mines is None:
      identified_mines = 0

    mine_prob = (n_mines - identified_mines) / plays_len

    X = calculated_mine_prob_matrix(X, identified_mines, mine_prob)

    prob = self.m_Model.predict_proba(X)

    prob[contains_zero] = numpy.array( [1, 0] )    
    prob[mine_rule_one] = numpy.array( [0, 1] )
    prob[mine_rule_two] = numpy.array( [0, 1] )
    prob[mine_rule_three] = numpy.array( [0, 1] )
    prob[contains_no_info] = numpy.array( [1 - mine_prob, mine_prob] )
    
    min_index = prob[:, 1].argmin()
    # print('calculated prob and identified mines:', X[min_index][-2:])
    # print('prob: ', prob[:, 1][min_index])
    return min_index, X
  # end def

  '''
  '''
  def report( self, i, j, n ):
    super( ).report( i, j, n )
  # end def

  '''
  '''
  def train_model(self, path_data, fname):
    data = numpy.loadtxt(path_data)
    X_t, X_v, y_t, y_v = load_data(data)

    model = Model(max_iter=500)
    new_model = model.fit(X_t, y_t)
    
    new_model_pred = new_model.predict(X_v)
    rec_new = recall_score(y_v, new_model_pred)
    print('rec_new: ', rec_new)

    old_model_pred = self.m_Model.predict(X_v)
    rec_old = recall_score(y_v, old_model_pred)
    print('rec_old: ', rec_old)

    if rec_new > rec_old + 0.1:
      joblib.dump(new_model, fname)
      
    return None
  # end def

# end class

## eof - Logistic.py
