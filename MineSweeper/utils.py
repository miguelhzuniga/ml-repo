from sklearn.model_selection import train_test_split
import numpy

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def load_data(data):
    X = data[:, :-1]
    contains_zero = (X[:, :8] == 0).any(axis=1)
    X = X[~ contains_zero]

    contains_no_info = (numpy.sum(X[:, :8] == -1, axis=1) + numpy.sum(X[:, :8] == 9, axis=1)) == 8
    X = X[~ contains_no_info]

    contains_ones = ((X[:, :8] >= 1) & (X[:, :8] != 9)).all(axis=1)
    X = X[~ contains_ones]

    mine_rule_two = (
        (numpy.sum(X[:, :8] == -1, axis=1) == 3)
        & (numpy.sum(X[:, :8] == 1, axis=1) == 5)
        )
    X = X[~ mine_rule_two]

    mine_rule_three = (
        (numpy.sum(X[:, :8] == -1, axis=1) == 5)
        & (numpy.sum((X[:, :8] >= 1) & (X[:, :8] != 9), axis=1) >= 2)
        )
    X = X[~ mine_rule_three]

    # X[:, 8] = sigmoid(X[:, 8])


    y = data[:, -1].astype(int)
    y = y[~ contains_zero]
    y = y[~ contains_no_info]
    y = y[~ contains_ones]
    y = y[~ mine_rule_two]
    y = y[~ mine_rule_three]


    X_t, X_v, y_t, y_v = train_test_split(
        X, y, test_size=0.3, random_state=2, 
        shuffle=True, stratify=y
    )

    return X_t, X_v, y_t, y_v

def calculated_mine_prob_matrix(matrix, identified_mines, mine_prob):
    # Count how many -1 values per row
    n_adj_cells = 8 - numpy.sum(matrix == -1, axis=1)  # shape: (n,)

    # Normalize each row by its n_adj_cells
    norm_matrix = matrix / n_adj_cells[:, None]  # broadcasting (n, 1)

    # Create a mask for valid values (not -1 and not 9)
    valid_mask = (matrix != -1) & (matrix != 9)

    # Compute the row-wise sum of valid values
    row_sums = (numpy.sum(norm_matrix * valid_mask, axis=1, keepdims=True)) + mine_prob

    # Create column filled with the constant identified_mines
    identified_mines_col = numpy.full((matrix.shape[0], 1), identified_mines)

    # Concatenate the original matrix with the row sums
    result = numpy.hstack((matrix, row_sums, identified_mines_col))
    
    result[:, 8] = sigmoid(result[:, 8])

    return result