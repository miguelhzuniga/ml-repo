def detTT(matrix:[list])->float:
  firstDiagonal = matrix[0][0] * matrix[1][1]
  secondDiagonal = matrix[0][1] * matrix[1][0]
  detMatrix = firstDiagonal - secondDiagonal
  return detMatrix


def adf(matrix:[list])->[list]:
  adjMtrix = None

  matrixp = [matrix[i][1:] * matrix[i][1] for i in range(len(matrix))]
  matrixpa=[(matrixp[:i] + matrixp[i+1:]) for i in range(len(matrixp))]

  return adjMatrix


def det(matrix:[list])->float:
  detMatrix = None
  return detMatrix


"""main"""
if __name__ == '__main__':

  import random


  a = [ random.randint(0,10) for i in range(4) ]
  b = [ random.randint(0,10) for i in range(4) ]
  c = [ random.randint(0,10) for i in range(4) ]
  d = [ random.randint(0,10) for i in range(4) ]
  matrix = [a, b, c, d]
  print(matrix)
