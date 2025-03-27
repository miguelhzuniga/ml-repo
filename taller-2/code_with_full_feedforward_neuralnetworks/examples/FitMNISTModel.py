## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import sys
sys.path.append( '../lib' )
import PUJ_ML

if __name__ == '__main__':

  # Parse command line arguments
  args = PUJ_ML.Helpers.ParseFitArguments(
      sys.argv,
      mandatory = [ ( 'dirname', str ), ( 'model', str ) ]
      )

  # Read model template
  model = PUJ_ML.Model.NeuralNetwork.FeedForward( )
  model.load( args.model )

  # Read data
  D_tr, D_te = PUJ_ML.IO.ReadMNIST( args.dirname )

  # print((D_tr[1].sum(axis=0)*100)/D_tr[1].shape[0])
  # print((D_te[1].sum(axis=0)*100)/D_te[1].shape[0])

  # Read model template
  model = PUJ_ML.Model.NeuralNetwork.FeedForward( )
  model.load( args.model )
  print( '==============================================' )
  print( 'Initial model: ' + str( model ) )
  print( '  ---> Encoded model: ' + model.encode64( ) )

  # Fit model
  PUJ_ML.Helpers.FitModel( model, args, D_tr, D_te )
  
  # Show final models
  print( '==============================================' )
  print( 'Fitted model: ' + str( model ) )
  print( '  ---> Encoded model: ' + model.encode64( ) )

  # Show final costs
  print( '==============================================' )
  print( 'Training cost = ' + str( model.cost( D_tr[ 0 ], D_tr[ 1 ] ) ) )
  if not D_te[ 0 ] is None:
    print( 'Testing cost  = ' + str( model.cost( D_te[ 0 ], D_te[ 1 ] ) ) )
  # end if

  # Compute Confusion matrices
  K_tr = PUJ_ML.Helpers.Confusion( model, D_tr[ 0 ], D_tr[ 1 ] )
  print( '==============================================' )
  print( 'Training confusion =\n', K_tr )
  if not D_te[ 0 ] is None:
    K_te = PUJ_ML.Helpers.Confusion( model, D_te[ 0 ], D_te[ 1 ] )
    print( 'Testing Confusion =\n', K_te )
  # end if

  # ROC curves
  ROC_tr = PUJ_ML.Helpers.ROC( model, D_tr[ 0 ], D_tr[ 1 ] )
  ROC_te = None
  if not D_te[ 0 ] is None:
    ROC_te = PUJ_ML.Helpers.ROC( model, D_te[ 0 ], D_te[ 1 ] )

# end if

## eof - FitMNISTModel.py
