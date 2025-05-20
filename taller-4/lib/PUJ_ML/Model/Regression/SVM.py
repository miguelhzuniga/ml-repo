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
    def _evaluate( self, X, threshold = False ):
        return ( ( X @ self.m_P[ 1 : , 0 ] ) - self.m_P[ 0 , 0 ] ).T
    # end def

    '''
    '''
    def cost_gradient( self, X, y, L1, L2, L ):
        z = numpy.asarray( self( X ) )
        y = numpy.asarray( y )
        margin = ( y * z )
        mask = ( margin < 1 ).ravel()

        J = self.cost( X, y, L1, L2, L )
        regularization = self._regularization( L1, L2, True, L )
        print(mask.shape)

        G = numpy.zeros( self.m_P.shape )
        G[ 0 ] = ( y[ mask, None ] ).sum() / float( X.shape[ 0 ] )
        G[ 1 : ] = ( -y[ mask, None ].T @ X[ mask, : ] ).sum( axis = 0 ).T / float( X.shape[ 0 ] )
        G += regularization

        return ( J, G )
    # end def

    '''
    '''
    def cost( self, X, y, L1, L2, L ):
        z = numpy.asarray( self( X ) )
        y = numpy.asarray( y )
        margin = ( y * z )
        mask = margin < 1 
       
        J = ( 1.0 - margin[ mask ] ).sum( )
        J += self._regularization( L1, L2, False, L ).sum()
        J /= float( X.shape[ 0 ] )

        return J
    # end def
# end class
## eof - SVM.py