## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import importlib.util, sys, numpy
from Board import *

'''
'''
def ImportLibrary( module_name, filename ):
  spec = importlib.util.spec_from_file_location( module_name, filename )
  if spec is None:
    print( f'Error: Could not create module specification for {filename}' )
    return None
  # end if
  module = importlib.util.module_from_spec( spec )
  sys.modules[ module_name ] = module
  try:
    spec.loader.exec_module( module )
    return module
  except Exception as e:
    print( f'Error executing module {filename}: {e}' )
    del sys.modules[ module_name ]
  # end try
  return None
# end def

"""
"""
if __name__ == '__main__':

  if len( sys.argv ) < 6:
    print(
      "Usage: python3", sys.argv[ 0 ], "width height mines player <arguments>"
      )
    sys.exit( 1 )
  # end if
  w = int( sys.argv[ 1 ] )
  h = int( sys.argv[ 2 ] )
  m = int( sys.argv[ 3 ] )
  player_fname = sys.argv[ 4 ]
  save_games_info = bool( int( sys.argv[ 7 ] ) )
  tr_model = bool( int( sys.argv[ 8 ] ) )
  try:
    trial_name = str(sys.argv[ 9 ])
  except:
    trial_name = None

  # Load player
  player_lib = ImportLibrary( 'Player', player_fname )
  player = player_lib.Player( sys.argv[ 5 : ] )

  # Load past games data
  player_name = player_fname.split( '/' )[ 1 ].split( '.' )[ 0 ]

  record_info = (player_name != 'Human') & save_games_info
  data_path = "Data/" + player_name + ".txt"
  pkl_path = "Data/" + player_name + ".pkl"
  trial_path = "Data/" + player_name + "_trials" + ".txt"

  if record_info:

    games_data = open( data_path, "r" )
    data = [ l.strip( ).split( ) for l in games_data.readlines( ) ]
    games_data.close( )

  if tr_model:
    player.train_model(data_path, pkl_path)

  if trial_name:
    trials_data = open( trial_path, "r" )
    trial_data = [ l.strip( ).split( ) for l in trials_data.readlines( ) ]
    trials_data.close( )


  trial = []
  # Create board
  board = Board( w, h, m )

  # Play!
  while not board.have_finished( ):
    print( board )
    choose_cell = player.choose_cell( w, h, m )

    if player_name != 'Human':
      i, j = choose_cell[0]
      X = choose_cell[1]
      movs = choose_cell[2]
    else:
      i, j = choose_cell

    print( 'Cell =', i, j )
    n = board.click( i, j )

    if record_info:
      y = board.have_lose()
      line = numpy.append(X, y)

      data.append(line)

    if trial_name:
      trial = [movs, trial_name]

    player.report( i, j, n )

  # end while

  if trial_name:
    trial_data.append(trial)

  # write game data
  if record_info:
    with open(data_path, 'w') as file:
      for arr in data:
          line = ' '.join(map(str, arr))
          file.write(f"{ line }\n")
  
  if trial_name:
    with open(trial_path, 'w') as file:
      for arr in trial_data:
          line = ' '.join(map(str, arr))
          file.write(f"{ line }\n")

  print( '====================================================' )
  print( board )
  if board.have_won( ):
    print( "You won!" )
  elif board.have_lose( ):
    print( "You lose :-(" )
  # end if
  print( '====================================================' )

# end if

## eof - MineSweeper.py
