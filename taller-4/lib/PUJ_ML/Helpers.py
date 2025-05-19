## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, random

'''
'''
def SplitDataForBinaryLabeling( A, train_coeff ):

  idx_z = numpy.where( A[ : , -1 ] == -1 )[ 0 ].tolist( )
  idx_o = numpy.where( A[ : , -1 ] == 1 )[ 0 ].tolist( )
  random.shuffle( idx_z )
  random.shuffle( idx_o )

  n = min( len( idx_z ), len( idx_o ) )
  n_tr = int( float( n ) * train_coeff )

  idx_tr = idx_z[ : n_tr ] + idx_o[ : n_tr ]
  random.shuffle( idx_tr )

  D_tr = A[ idx_tr , : ]

  D_te = None
  if n_tr < n:
    idx_te = idx_z[ n_tr : n ] + idx_o[ n_tr : n ]
    random.shuffle( idx_te )
    D_te = A[ idx_te , : ]

  return ( D_tr, D_te )
  # end if
# end def

'''
'''
def Confussion( m, X, y ):
  try:
    z = m( X, True )
    z = numpy.where(z >= 1, 1, -1)
  except:
    z = m.predict(X)
  
  z = numpy.asarray(z).flatten()
  y = numpy.asarray(y).flatten()

  yp = numpy.zeros((z.shape[0], 2))
  yo = numpy.zeros((y.shape[0], 2))

  yp[z == -1, 0] = 1
  yp[z ==  1, 1] = 1
  yo[y == -1, 0] = 1
  yo[y ==  1, 1] = 1

  K = yo.T @ yp

  TP = float(K[1, 1])
  TN = float(K[0, 0])
  FN = float(K[1, 0])
  FP = float(K[0, 1])

  sensibility = TP / (TP + FN) if TP + FN != 0 else 0
  specificity = TN / (TN + FP) if TN + FP != 0 else 0
  accuracy = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN != 0 else 0
  F1 = 2 * TP / (2 * TP + FP + FN) if TP + (FP + FN) != 0 else 0

  return (K, sensibility, specificity, accuracy, F1)
# end def

'''
'''
def ROC( m, X, y ):

  y_true = y.T.tolist( )[ 0 ]
  y_scores = m( X ).T.tolist( )[ 0 ]
  D = sorted( zip( y_scores, y_true ), reverse = True )

  n_pos = sum( y_true )
  n_neg = len( y_true ) - n_pos

  fpr = [ 0 ]
  tpr = [ 0 ]

  tp = 0
  fp = 0

  for i in range( len( D ) ):
    score, true_label = D[ i ]

    if i == 0 or score != D[ i - 1 ][ 0 ]:
      fpr.append( fp / n_neg )
      tpr.append( tp / n_pos )
    # end if

    if true_label == 1:
      tp += 1
    else:
      fp += 1
    # end if
  # end for

  fpr.append( 1 )
  tpr.append( 1 )

  return ( fpr, tpr )
# end def

## eof - Helpers.py
