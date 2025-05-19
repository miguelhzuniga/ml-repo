## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import math, numpy

"""
"""
class Base:
    
  '''
  '''
  m_Model = None
  m_Lambda1 = 0
  m_Lambda2 = 0
  m_L = 0
  m_Epsilon = None
  m_Debug = None

  '''
  '''
  def __init__( self, m ):
    self.m_Model = m

    self.m_Epsilon = float( 1 )
    while self.m_Epsilon + 1 > 1:
      self.m_Epsilon /= 2
    # end while
    self.m_Epsilon *= 2
  # end def

  '''
  '''
  def fit( self, D_tr, L1, L2, L, D_te = None, validation = 'mce', K = 5, batch_size = 0 ):

    batches = []
    if validation.lower( ) != 'mce':

      # Blend all training and test data
      X_tr = D_tr[ : , : D_tr.shape[ 1 ] - 1 ]
      y_tr = numpy.asmatrix( D_tr[ : , -1 ] ).T
      if not D_te is None:
        X_tr = numpy.concatenate(
          ( X_tr, D_te[ : , : D_te.shape[ 1 ] - 1 ] ), axis = 0
          )
        y_tr = numpy.concatenate(
          ( y_tr, numpy.asmatrix( D_te[ : , -1 ] ).T ), axis = 0
          )
      # end if
      if validation.lower( ) == 'loo':
        self._fit_loo( X_tr, y_tr, batches, L1, L2, L )
      elif validation.lower( ) == 'kfold':
        self._fit_kfold( X_tr, y_tr, K, batches, L1, L2, L )
      else:
        raise ValueError( 'Validation not valid "' + validation + '"' )
      # end if
    else:

      all_indices = [ i for i in range( D_tr.shape[ 0 ] ) ]
      if batch_size == 0:
        batches = [ all_indices ]
      elif batch_size > 0:
        n_batches = int( math.ceil( float( D_tr.shape[ 0 ] ) / float( batch_size ) ) )

        s = 0
        for b in range( n_batches - 1 ):
          batches += [ all_indices[ s : s + batch_size ] ]
          s += batch_size
        # end for
        ## batches += [ all_indices[ s : s + len( all_indices ) - ( n_batches * batch_size ) ] ]

      # end if

      self._fit_mce( D_tr, D_te, batches )
    # end if
  # end def

  '''
  '''
  def _fit_mce( self, D_tr, D_te, batches ):

    X_tr, y_tr, X_te, y_te = None, None, None, None

    X_tr = D_tr[ : , : D_tr.shape[ 1 ] - 1 ]
    y_tr = numpy.asmatrix( D_tr[ : , -1 ] ).T
    if not D_te is None:
      X_te = D_te[ : , : D_te.shape[ 1 ] - 1 ]
      y_te = numpy.asmatrix( D_te[ : , -1 ] ).T
    # end if
    self._fit( X_tr, y_tr, X_te, y_te, batches )
  # end def

  '''
  '''
  def _fit_loo( self, X, y, batches, L1, L2, L ):

    M = X.shape[ 0 ]
    idx = [ i for i in range( M ) ]

    v = 0
    for m in range( M ):
      X_tr = X[ idx[ : m ] + idx[ m + 1 : ] , : ]
      y_tr = y[ idx[ : m ] + idx[ m + 1 : ] , : ]

      print( '*** Leave-one-out: ' + str( m ) + '/' + str( M - 1 ) )
      self._fit( X_tr, y_tr, None, None, batches )

      v += self.m_Model.cost( X[ m , : ], y[ m , : ], L1, L2, L )
    # end for
    v /= float( M )

    message = '*** Leave-one-out final validation value = ' + str( v ) + ' ***'
    margin = ''.join( [ '*' for i in range( len( message ) ) ] )
    print( margin )
    print( message )
    print( margin )
  # end def

  '''
  '''
  def _fit_kfold( self, X, y, K, batches, L1, L2, L ):

    M = X.shape[ 0 ]
    N = math.ceil( M / K )
    idx = []
    f = 0
    for k in range( K ):
      idx += [ [ f + i for i in range( N ) if ( f + i ) < M ] ]
      f += len( idx[ -1 ] )
    # end for

    v = 0
    for k in range( K ):
      idx_tr = sum( [ idx[ i ] for i in range( K ) if i != k ], [] )
      X_tr = X[ idx_tr , : ]
      y_tr = y[ idx_tr , : ]
      print( '*** Kfold (K=' + str( K ) + '): ' + str( k + 1 ) + '/' + str( K ) )
      self._fit( X_tr, y_tr, None, None, batches )

      v += self.m_Model.cost( X[ idx[ k ] , : ], y[ idx[ k ] , : ], L1, L2, L )
    # end for
    v /= float( M )

    message = \
      '*** Kfold (K=' + str( K ) \
      + \
      ') final validation value = ' \
      + \
      str( v ) \
      + \
      ' ***'
    margin = ''.join( [ '*' for i in range( len( message ) ) ] )
    print( margin )
    print( message )
    print( margin )
  # end def

  '''
  '''
  def _fit( self, X_tr, y_tr, X_te, y_te ):
    pass
  # end def

# end class

## eof - Base.py
