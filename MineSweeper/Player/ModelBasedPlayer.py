## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

from abc import ABC, abstractmethod
import numpy, random, time

"""
"""
class ModelBasedPlayer:

  '''
  '''
  m_Model     = None
  m_Plays     = None
  m_Marks     = None
  m_Width     = 0
  m_Height    = 0
  m_SleepTime = 0.2

  '''
  '''
  def __init__( self, args ):
    self.read_model( args[ 0 ] )
    if len( args ) > 1:
      self.m_SleepTime = float( args[ 1 ] )
    # end if
  # end def

  '''
  '''
  def choose_cell( self, w, h, n ):

    # Init game state
    if self.m_Marks is None:
      self.m_Marks = [ [ 9 for j in range( h ) ] for i in range( w ) ]
      self.m_Width = w
      self.m_Height = h

      self.m_Plays = []
      for i in range( w ):
        for j in range( h ):
          self.m_Plays += [ ( i, j ) ]
        # end for
      # end for
    # end if

    # Choose a play
    random.shuffle( self.m_Plays )
    dX = []
    for p in self.m_Plays:
      for k in range( -1, 2 ):
        for l in range( -1, 2 ):
          if k != 0 or l != 0:
            i = p[ 0 ] + k
            j = p[ 1 ] + l
            if 0 <= i and i < self.m_Width and 0 <= j and j < self.m_Height:
              dX += [ self.m_Marks[ i ][ j ] ]
            else:
              dX += [ -1 ]
            # end if
          # end if
        # end for
      # end for
    # end for

    X = numpy.array( dX ).reshape( ( len( self.m_Plays ), 8 ) )
    evaluated = self.evaluate( X,  len( self.m_Plays ), n)
    y = evaluated[0]
    X_t = evaluated[1][y]
    o = self.m_Plays[ y ]
    self.m_Plays.pop( y )
    movs = self.m_Height * self.m_Width - len(self.m_Plays)

    return (o, X_t, movs)
  # end def

  '''
  '''
  @abstractmethod
  def report( self, i, j, n ):
    self.m_Marks[ i ][ j ] = n
    time.sleep( self.m_SleepTime )
  # end def

  '''
  '''
  @abstractmethod
  def read_model( self, fname ):
    pass
  # end def

  '''
  '''
  @abstractmethod
  def evaluate( self, X ):
    pass
  # end def
# end class

## eof - ModelBasedPlayer.py
