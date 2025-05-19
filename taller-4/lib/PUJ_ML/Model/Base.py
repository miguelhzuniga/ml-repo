## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy

"""
"""
class Base:

  '''
  '''
  m_P = None

  '''
  '''
  def __init__( self, n = 1 ):
    self.m_P = numpy.zeros( ( n, 1 ) )
  # end def
  
  '''
  '''
  def __call__( self, X, threshold = False ):
    if not self.m_P is None:
      if isinstance( X, numpy.matrix ):
        return self._evaluate( X )
      elif isinstance( X, numpy.ndarray ):
        return self( numpy.asmatrix( X ) )
      elif isinstance( X, list ):
        m = self.input_size( )
        s = ( len( X ) // m, m )
        n = s[ 0 ] * s[ 1 ]
        return self( numpy.reshape( numpy.matrix( X[ : n ] ), newshape = s ) )
      else:
        return None
      # end if
    else:
      return None
    # end if
  # end def
  
  '''
  '''
  def __getitem__( self, i ):
    if not self.m_P is None:
      if i < self.m_P.shape[ 0 ]:
        return self.m_P[ i, 0 ]
      else:
        return float( 0 )
      # end if
    else:
      return float( 0 )
    # end if
  # end def

  '''
  '''
  def __setitem__( self, i, v ):
    if not self.m_P is None:
      if i < self.m_P.shape[ 0 ]:
        self.m_P[ i, 0 ] = v
      # end if
    # end if
  # end def

  '''
  '''
  def __iadd__( self, w ):
    self.m_P += w
    return self
  # end def

  '''
  '''
  def __isub__( self, w ):
    self.m_P -= w
    return self
  # end def
  
  '''
  '''
  def __str__( self ):
    s = ''
    if not self.m_P is None:
      n = self.m_P.shape[ 0 ]
      s = str( n )
      for i in range( n ):
        s += ' {v:.4f}'.format( v = self.m_P[ i , 0 ] )
      # end for
    else:
      s = '0'
    # end if
    return s
  # end def
  
  '''
  '''
  def input_size( self ):
    return self.size( )
  # end def

  '''
  '''
  def size( self ):
    return self.m_P.shape[ 0 ]
  # end def

  '''
  '''
  def init( self ):
    if not self.m_P is None:
      self.m_P *= float( 0 )
    # end if
  # end def

  '''
  '''
  def _regularization( self, L1, L2, Derivate, L = 1 ):
    assert abs(L1 + L2 - 1.0) < 1e-8, "L1 and L2 must sum to 1"

    if Derivate:
      r = self.m_P * ( float( 2 ) * L2 )
      pi = numpy.where( self.m_P > 0 )[ 0 ]
      ni = numpy.where( self.m_P < 0 )[ 0 ]
      if pi.size > 0: r[ pi ] += L1
      if ni.size > 0: r[ ni ] -= L1
      return numpy.asmatrix( r * L )
    
    else:
      r = ( self.m_P ** 2 ) * L2
      pi = numpy.where( self.m_P >= 0 )[ 0 ]
      ni = numpy.where( self.m_P < 0 )[ 0 ]
      if pi.size > 0: r[ pi ] += L1 * self.m_P[ pi ]
      if ni.size > 0: r[ ni ] -= L1 * self.m_P[ pi ]
      return numpy.asmatrix( r * L )
  # end def
# end class

## eof - Base.py
