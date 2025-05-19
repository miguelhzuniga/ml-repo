## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy
from .Linear import Linear

"""
"""
class SVM( Linear ):

    '''
    '''
    m_Epsilon = 0

    '''
    '''
    def __init__( self, n = 1 ):
        super( ).__init__( n )

        self.m_Epsilon = float( 1 )
        while self.m_Epsilon + 1 > 1:
            self.m_Epsilon /= 2
        # end while
        self.m_Epsilon *= 2
        
    # end def

    '''
    '''
    def _evaluate( self, X ):
        return super( )._evaluate( X )
    # end def

    '''
    '''
    def cost_gradient( self, X, y, L1, L2, L ):
        z = self( X )
        J = self.cost( X, y, L1, L2, L )
        oi = numpy.where( z < 1 )[ 0 ].tolist( )

        G = numpy.zeros( self.m_P.shape )
        G[ 0 ] = ( y[ oi ] ).sum() / float( X.shape[ 0 ] )
        G[ 1 : ] = numpy.multiply( X[ oi , : ], -y[ oi ] ).sum( axis = 0 ).T / float( X.shape[ 0 ] )
        G = G + self._regularization( L1, L2, L )

        return ( J, G )
    # end def

    '''
    '''
    def cost( self, X, y, L1, L2, L ):
        z = self( X )
        # zi = numpy.where( y >= 1 )[ 0 ].tolist( )
        oi = numpy.where( z < 1 )[ 0 ].tolist( )
        
        J = ( float( 1 ) - ( y[ oi ].T @ z[ oi , : ] ) ).sum( )
        J += self._regularization( L1, L2, L ).sum()
        J /= float( X.shape[ 0 ] )

        return J
    # end def
# end class
## eof - SVM.py