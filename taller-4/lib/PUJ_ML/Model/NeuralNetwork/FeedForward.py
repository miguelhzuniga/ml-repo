## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, random
from ..Base import Base

"""
"""
class FeedForward( Base ):

  '''
  '''
  m_W = []
  m_B = []
  m_S = []
  m_Epsilon = 0


  '''
  '''
  def __init__( self, n = 0 ):
    super( ).__init__( 0 )

    self.m_Epsilon = float( 1 )
    while self.m_Epsilon + 1 > 1:
      self.m_Epsilon /= 2
    # end while
    self.m_Epsilon *= 2

  # end def

  '''
  '''
  def __getitem__( self, l, i, j ):
    if l < self.number_of_layers( ):
      return self.m_W[ l ][ i , j ]
    else:
      return 0
    # end if
  # end def

  '''
  '''
  def __getitem__( self, l, i ):
    if l < self.number_of_layers( ):
      return self.m_B[ l ][ 0 , 1 ]
    else:
      return 0
    # end if
  # end def

  '''
  '''
  def __setitem__( self, l, i, j, v ):
    if l < self.number_of_layers( ):
      self.m_W[ l ][ i , j ] = v
    # end if
  # end def

  '''
  '''
  def __setitem__( self, l, i, v ):
    if l < self.number_of_layers( ):
      self.m_B[ l ][ 0 , 1 ] = v
    # end if
  # end def

  '''
  '''
  def __iadd__( self, w ):
    L = self.number_of_layers( )
    k = 0
    for l in range( L ):
      in_size = self.m_W[ l ].shape[ 0 ]
      out_size = self.m_W[ l ].shape[ 1 ]
      for i in range( in_size ):
        for o in range( out_size ):
          self.m_W[ l ][ i , o ] += w[ k , 0 ]
          k += 1
        # end for
      # end for
      for o in range( out_size ):
        self.m_B[ l ][ 0 , o ] += w[ k , 0 ]
        k += 1
      # end for
    # end for
    return self
  # end def

  '''
  '''
  def __isub__( self, w ):
    L = self.number_of_layers( )
    k = 0
    for l in range( L ):
      in_size = self.m_W[ l ].shape[ 0 ]
      out_size = self.m_W[ l ].shape[ 1 ]
      for i in range( in_size ):
        for o in range( out_size ):
          self.m_W[ l ][ i , o ] -= w[ k , 0 ]
          k += 1
        # end for
      # end for
      for o in range( out_size ):
        self.m_B[ l ][ 0 , o ] -= w[ k , 0 ]
        k += 1
      # end for
    # end for
    return self
  # end def

  '''
  '''
  def __str__( self ):
    h = '0'
    p = ''

    for l in range( self.number_of_layers( ) ):
      in_size = self.m_W[ l ].shape[ 0 ]
      out_size = self.m_W[ l ].shape[ 1 ]

      if l == 0:
        h = str( in_size ) + '\n'
      # end if
      h += str( out_size ) + ' ' + str( self.m_S[ l ] ) + '\n'

      for i in range( in_size ):
        for o in range( out_size ):
          p += str( self.m_W[ l ][ i, o ] ) + ' '
        # end for
      # end for
      for o in range( out_size ):
        p += str( self.m_B[ l ][ 0 , o ] ) + ' '
      # end for

    # end for

    return h + p
  # end def

  '''
  '''
  def size( self ):
    s = 0
    for l in range( self.number_of_layers( ) ):
      s += self.m_W[ l ].shape[ 0 ] * self.m_W[ l ].shape[ 1 ]
      s += self.m_B[ l ].shape[ 0 ] * self.m_B[ l ].shape[ 1 ]
    # end for
    return s
  # end def

  '''
  '''
  def add_input_layer( self, in_size, out_size, activation ):
    self.m_W = [ numpy.zeros( ( in_size, out_size ) ) ]
    self.m_B = [ numpy.zeros( ( 1, out_size ) ) ]
    self.m_S = [ activation ]
  # end def

  '''
  '''
  def add_layer( self, out_size, activation ):
    if self.number_of_layers( ) > 0:
      in_size = self.m_B[ -1 ].shape[ 1 ]
      self.m_W += [ numpy.zeros( ( in_size, out_size ) ) ]
      self.m_B += [ numpy.zeros( ( 1, out_size ) ) ]
      self.m_S += [ activation ]
    else:
      raise AssertionError( 'Input layer not yet defined.' )
    # end if
  # end def

  '''
  '''
  def init( self ):
    for l in range( self.number_of_layers( ) ):
      in_size = self.m_W[ l ].shape[ 0 ]
      out_size = self.m_W[ l ].shape[ 1 ]

      for i in range( in_size ):
        for o in range( out_size ):
          self.m_W[ l ][ i , o ] = random.random( ) - 0.5
        # end for
      # end for
      for o in range( out_size ):
        self.m_B[ l ][ 0 , o ] = random.random( ) - 0.5
      # end for
    # end for
  # end def

  '''
  '''
  def input_size( self ):
    if self.number_of_layers( ) > 0:
      return self.m_W[ 0 ].shape[ 0 ]
    else:
      return 0
    # end if
  # end def
  '''
  '''

  def number_of_layers( self ):
    return len( self.m_W )
  # end def

  '''
  '''
  def cost_gradient( self, X, y, L1, L2 ):

    # Forward
    A = [ X ]
    Z = []
    L = self.number_of_layers( )
    for l in range( L ):
      Z += [ ( A[ -1 ] @ self.m_W[ l ] ) + self.m_B[ l ] ]
      A += [ self.m_S[ l ]( Z[ -1 ] ) ]
    # end for

    G = numpy.zeros( ( 1, self.size( ) ) )
    # Backward
    m = float( 1 ) / float( X.shape[ 0 ] )
    DL = A[ L ] - y
    GbL = DL.sum( axis = 0 ) * m
    GwL = ( A[ L - 1 ].T @ DL ) * m

    k = G.shape[ 1 ] - 1
    G[ 0 , k ] = GbL
    k -= 1
    h = k - GwL.size + 1
    for r in range( GwL.shape[ 0 ] ):
      for c in range( GwL.shape[ 1 ] ):
        G[ 0 , h ] = GwL[ r , c ]
        h += 1
      # end for
    # end for
    k -= GwL.size

    for l in range( L - 1, 0, -1 ):
      DL = numpy.multiply( ( DL @ self.m_W[ l ].T ), self.m_S[ l - 1 ]( Z[ l - 1 ], True ) )
      GBl = DL.sum( axis = 0 ) * m
      GWl = ( A[ l - 1 ].T @ DL ) * m

      h = k - GBl.size + 1
      for r in range( GBl.shape[ 0 ] ):
        for c in range( GBl.shape[ 1 ] ):
          G[ 0 , h ] = GBl[ r , c ]
          h += 1
        # end for
      # end for
      k -= GBl.size
      h = k - GWl.size + 1
      for r in range( GWl.shape[ 0 ] ):
        for c in range( GWl.shape[ 1 ] ):
          G[ 0 , h ] = GWl[ r , c ]
          h += 1
        # end for
      # end for
      k -= GWl.size
    # end for

    zi = numpy.where( y == 0 )[ 0 ].tolist( )
    oi = numpy.where( y == 1 )[ 0 ].tolist( )

    J  = numpy.log( float( 1 ) - A[ -1 ][ zi , : ] + self.m_Epsilon ).sum( )
    J += numpy.log( A[ -1 ][ oi , : ] + self.m_Epsilon ).sum( )
    J /= float( X.shape[ 0 ] )

    return ( -J, G )
  # end def

  '''
  '''
  def cost( self, X, y ):
    z = self( X )

    zi = numpy.where( y == 0 )[ 0 ].tolist( )
    oi = numpy.where( y == 1 )[ 0 ].tolist( )

    J  = numpy.log( float( 1 ) - z[ zi , : ] + self.m_Epsilon ).sum( )
    J += numpy.log( z[ oi , : ] + self.m_Epsilon ).sum( )
    J /= float( X.shape[ 0 ] )

    return -J
  # end def

  '''
  '''
  def _evaluate( self, X, threshold ):
    L = self.number_of_layers( )
    if L > 0:
      a = X
      for l in range( L ):
        z = ( a @ self.m_W[ l ] ) + self.m_B[ l ]
        a = self.m_S[ l ]( z )
      # end for
      return a
    else:
      return None
    # end if
  # end def

  '''
  '''
  def _regularization( self, L1, L2 ):
    r = numpy.zeros( ( self.size( ), 1 ) )

    if L1 != float( 0 ) or L2 != float( 0 ):
      _L2 = float( 2 ) * L2
      L = self.number_of_layers( )
      k = 0
      for l in range( L ):
        in_size = self.m_W[ l ].shape[ 0 ]
        out_size = self.m_W[ l ].shape[ 1 ]
        for i in range( in_size ):
          for o in range( out_size ):
            r[ k , 0 ] = self.m_W[ l ][ i , o ] * _L2
            if self.m_W[ l ][ i , o ] > 0:
              r[ k , 0 ] += L1
            elif self.m_W[ l ][ i , o ] < 0:
              r[ k , 0 ] -= L1
            # end if
            k += 1
          # end for
        # end for
        for o in range( out_size ):
          r[ k , 0 ] = self.m_B[ l ][ 0 , o ] * _L2
          if self.m_B[ l ][ 0 , o ] > 0:
            r[ k , 0 ] += L1
          elif self.m_B[ l ][ 0 , o ] < 0:
            r[ k , 0 ] -= L1
          # end if
          k += 1
        # end for
      # end for
    # end if
    return numpy.asmatrix( r )
  # end def

# end class

## eof - FeedForward.py
