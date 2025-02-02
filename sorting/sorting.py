## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================
## Comment
import time

'''
Sorts a sequence of comparable (<) elements
@input  S a reference to a secuence of comparable elements.
@output S an ordered permutation of the input.
@complexity O(|S|^2)
'''
def Bubble( S ):
  for j in range( len( S ) ):
    for i in range( len( S ) - 1 ):
      if S[ i + 1 ] < S[ i ]:
        S[ i ], S[ i + 1 ] = S[ i + 1 ], S[ i ]
      # end if
    # end for
  # end for
# end def

'''
Sorts a sequence of comparable (<) elements
@input  S a reference to a secuence of comparable elements.
@output S an ordered permutation of the input.
@complexity O(|S|^2), Om(|S|)
'''
def Insertion( S ):
  for j in range( len( S ) ):
    k = S[ j ]
    i = j - 1
    while 0 <= i and k < S[ i ]:
      S[ i + 1 ] = S[ i ]
      i = i - 1
    # end while
    S[ i + 1 ] = k
  # end for
# end def

'''
Sorts a sequence of comparable (<) elements
@input    S a reference to a secuence of comparable elements.
@optional b first index
@optional e one past last index
@output   S an ordered permutation of the input.
@complexity Th(|S|.log2(|S|))
'''
def Merge( S, b = -1, e = -1 ):
  r = e
  if e < 0:
    r = len( S ) - 1
  # end if
  if b < 0:
    Merge( S, 0, r )
  else:
    if b < r:

      # Pivot
      q = ( b + r ) // 2

      # Recursive problems
      Merge( S, b, q )
      Merge( S, q + 1, r )

      # Merge initialization
      L = S[ b : q + 1 ]
      R = S[ q + 1 : r + 1 ]
      i, j, k = 0, 0, b

      # Merge L and R
      while i < len( L ) and j < len( R ):
        if L[ i ] < R[ j ]:
          S[ k ] = L[ i ]
          i += 1
        else:
          S[ k ] = R[ j ]
          j += 1
        # end if
        k += 1
      # end while

      # Merge remaining L
      while i < len( L ):
        S[ k ] = L[ i ]
        i += 1
        k += 1
      # end while

      # Merge remaining R
      while j < len( R ):
        S[ k ] = R[ j ]
        j += 1
        k += 1
      # end while

    # end if
  # end if
# end def

'''
'''
def IsSorted( S ):
  f = True
  for i in range( len( S ) - 1 ):
    f = f and not S[ i ] > S[ i + 1 ]
  # end for
  return f
# end def

'''
Execute a sorting algorithm A over a sequence S and measure its time
'''
def Exec( A, S ):
  C = S.copy( )

  st = time.time( )
  A( C )
  et = time.time( )

  if not IsSorted( C ):
    raise Exception( 'Sequence not sorted.', 'Sequence not sorted.' )
  # end if

  return et - st

# end def

"""
========================================================
************************* MAIN *************************
========================================================
"""
if __name__ == '__main__':

  import random
  import sys
  import pandas as pd
  import matplotlib.pyplot as plt

  sequenceLength = []
  bubble = []
  insertion = []
  merge = []

  #get step and stop
  step = int(input("Write a integer that defines the step of the sequence\n"))
  stop = int(input("Write a integer that defines the stop of the sequence\n"))

  if ((step>0) & (stop>0) & ((stop%step)==0)):
    sequencesLength = range(step, stop, step)
  else:
    print("Invalid step or stop values")
    sys.exit()
  # end if

  for s in sequencesLength:
    sequence = [ random.randint( int(-1e5), int(1e5) ) for i in range( s ) ]

    tb = Exec( Bubble, sequence )
    ti = Exec( Insertion, sequence )
    tm = Exec( Merge, sequence )

    sequenceLength.append(s)
    bubble.append(tb)
    insertion.append(ti)
    merge.append(tm)

  data = {
    "Length": sequenceLength,
    "Bubble": bubble,
    "Insertion": insertion,
    "Merge": merge
  }

  df = pd.DataFrame(data)

  print(df.dtypes)
  print(df)

  # Plot
  plt.figure(figsize=(10, 6))
  plt.plot(df["Length"], df["Bubble"], label="Bubble Sort")
  plt.plot(df["Length"], df["Insertion"], label="Insertion Sort")
  plt.plot(df["Length"], df["Merge"], label="Merge Sort")

  # Labels and Title
  plt.xlabel("Length")
  plt.ylabel("Time (seconds)")
  plt.title("Sorting Algorithm Performance")
  plt.legend()
  plt.grid(True)

  # Show plot
  plt.show()
# end if

## eof - sorting.py
