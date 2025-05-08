## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import sys
import tensorflow

if __name__ == '__main__':

  layer_class = tensorflow.keras.layers.Dense
  layers  = [ layer_class( 32, activation = 'relu', input_shape = ( 8, ) ) ]
  layers += [ layer_class( 16, activation = 'relu' ) ]
  layers += [ layer_class( 8, activation = 'relu' ) ]
  layers += [ layer_class( 1, activation = 'sigmoid' ) ]

  model = tensorflow.keras.models.Sequential( layers )
  model\
    .compile( \
      optimizer = 'adam',\
      loss = 'sparse_categorical_crossentropy', \
      metrics = [ 'accuracy' ] \
      )
  model.save( sys.argv[ 1 ] )
# end if

## eof - CreateFeedForwardNeuralNetwork.py
