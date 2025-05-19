## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy

class ActivationFunctions:

  '''
  '''
  def get( name ):
    if name.lower( ) == 'identity':
      return ActivationFunctions.Identity
    elif name.lower( ) == 'relu':
      return ActivationFunctions.ReLU
    elif name.lower( ) == 'sigmoid':
      return ActivationFunctions.Sigmoid
    elif name.lower( ) == 'tanh':
      return ActivationFunctions.Tanh
    else:
      return None
    # end if
  # end def

  '''
  '''
  def Identity( Z, d = False ):
    if d:
      return numpy.ones( Z.shape )
    else:
      return Z
    # end if
  # end def

  '''
  '''
  def ReLU( Z, d = False ):
    if d:
      return ( Z > 0 ).astype( float )
    else:
      return numpy.multiply( Z, ( Z > 0 ).astype( float ) )
    # end if
  # end def

  '''
  '''
  def Sigmoid( Z, d = False ):
    if d:
      s = Sigmoid( Z, False )
      return numpy.multiply( z, float( 1 ) - z )
    else:
      return float( 1 ) / ( float( 1 ) + numpy.exp( -Z ) )
    # end if
  # end def

  '''
  '''
  def Tanh( Z, d = False ):
    if d:
      T = Tanh( Z, False )
      return float( 1 ) - numpy.multiply( T, T )
    else:
      numpy.tanh( Z )
    # end if
  # end def
    
# end class

## eof - ActivationFunctions.py









