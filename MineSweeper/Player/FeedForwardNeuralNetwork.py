## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import itertools, numpy, os, sys
import tensorflow
sys.path.append( os.path.dirname( os.path.abspath( __file__ ) ) )
from ModelBasedPlayer import ModelBasedPlayer

"""
"""
class Player( ModelBasedPlayer ):

  '''
  '''
  def __init__( self, args ):
    super( ).__init__( args )
  # end def

  '''
  '''
  def read_model( self, fname ):
    self.m_Model = tensorflow.keras.models.load_model( fname )
  # end def

  '''
  '''
  def evaluate( self, X ):
    return self.m_Model.predict( X ).argmin( )
  # end def
# end class

## eof - FeedForwardNeuralNetwork.py
