## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import argparse
import sys
import numpy
sys.path.append( '../lib' )
from PUJ_ML.Helpers import SplitDataForBinaryLabeling, Confussion
from PUJ_ML.Model.Regression.SVM import SVM
from PUJ_ML.Optimizer.Adam import Adam
from PUJ_ML.Optimizer.GradientDescent import GradientDescent
from PUJ_ML.GraphicConfussionDebugger import GraphicConfussionDebugger
from PUJ_ML.ConfussionDebugger import ConfussionDebugger
import pandas
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

'''
'''
def argument_parser():
    # -- Parse command line arguments
    parser = argparse.ArgumentParser(
        prog = sys.argv[ 0 ],
        description = 'Fit a logistic regression',
        epilog = 'Enjoy?'
        )
    parser.add_argument( 'data', type = str )
    parser.add_argument( '-test', '--test', type = str, default = '0' )
    parser.add_argument(
        '-d', '--delimiter', type = str, default = ' ',
        help = 'Delimiter in the CSV file'
    )
    parser.add_argument(
        '-o', '--optimizer', type = str, default = 'Adam',
        help = 'Adam|GradientDescent'
    )
    parser.add_argument(
        '-debugger', '--debugger', type = str,
        default = 'Simple', help = 'Simple|Graphic'
    )
    parser.add_argument(
        '-v', '--validation', type = str, default = 'MCE',
        help = 'MCE|LOO|KfoldK'
    )
    parser.add_argument( '-a', '--alpha', type = float, default = 1e-2 )
    parser.add_argument( '-L1', '--L1', type = float, default = 0 )
    parser.add_argument( '-L2', '--L2', type = float, default = 0 )
    parser.add_argument( '-L', '--L', type = float, default = 1 )
    parser.add_argument( '-e', '--epochs', type = int, default = 1000 )
    try:
        args = parser.parse_args( )
        return args
    except BaseException as error:
        sys.exit( 1 )
    # end try
# end def

'''
'''
def load_data(data, delimiter):
    dataset = pandas.read_csv(data, sep=delimiter)
    dataset = dataset[
        [
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
            'smoothness_mean', 'compactness_mean', 'concavity_mean',
            'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
            'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
            'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
            'fractal_dimension_se', 'radius_worst', 'texture_worst',
            'perimeter_worst', 'area_worst', 'smoothness_worst',
            'compactness_worst', 'concavity_worst', 'concave points_worst',
            'symmetry_worst', 'fractal_dimension_worst', 'diagnosis'
        ]
    ].copy()
    dataset['diagnosis'] = pandas.to_numeric(
        numpy.where(dataset['diagnosis'] == 'M', '1', '-1')
    )

    # cols_to_scale = [col for col in dataset.columns if col != 'diagnosis']
    # dataset[cols_to_scale] = (
    #     (dataset[cols_to_scale] - dataset[cols_to_scale].mean())
    #     / dataset[cols_to_scale].std()
    # )

    return dataset.to_numpy()
# end def

'''
'''
def prepare_data(A, test, split_x_y = True):
    # -- Prepare data
    D_tr = None
    D_te = None
    
    try:
        train_coeff = abs( float( 1 ) - abs( float( test ) ) )
        if train_coeff > 1: train_coeff = 1
        elif train_coeff < 1:
            D_tr, D_te = SplitDataForBinaryLabeling( A, train_coeff )
        else:
            D_tr = A
        # end if
    except ValueError:
        print(ValueError)
        sys.exit( 1 )
    # end try
    
    # -- Check sizes
    if not D_te is None:
        if D_tr.shape[ 1 ] != D_te.shape[ 1 ]:
            print( 'Data sizes are not compatible.' )
            sys.exit( 1 )
        # end if
    # end if

    # if split_x_y:
    #     y_tr = D_tr[ : , -1 ]
    #     y_te = D_te[ : , -1 ]
    #     X_tr = D_tr[ : , : -1 ]
    #     X_te = D_te[ : , : -1 ]
    #     return ( X_tr, y_tr, X_te, y_te )
    # else:
    return ( D_tr, D_te )
    # end if
# end def


if __name__ == "__main__":

    args = argument_parser()
    A = load_data( args.data, args.delimiter )
    D_tr, D_te = prepare_data( A, args.test )
    
    m = SVM( D_tr.shape[ 1 ] - 1 )
    print( 'Initial model: ' + str( m ) )

    # -- Prepare optimizer
    opt = None
    if args.optimizer.lower( ) == 'adam':
        opt = Adam( m )
    elif args.optimizer.lower( ) == 'gradientdescent':
        opt = GradientDescent( m )
    # end if
    if opt is None:
        print( 'Invalid optimizer "' + args.optimizer + '"' )
        sys.exit( 1 )
    # end if

    opt.m_Alpha = args.alpha
    opt.m_Lambda1 = args.L1
    opt.m_Lambda2 = args.L2
    opt.m_L = args.L2
    if args.debugger.lower( ) == 'simple':
        opt.m_Debug = ConfussionDebugger( args.epochs, m, D_tr, D_te )
    elif args.debugger.lower( ) == 'graphic':
        opt.m_Debug = GraphicConfussionDebugger( args.epochs, m, D_tr, D_te )
    # end if

    # -- Fit model to train data
    if args.validation.lower( ) == 'mce':
        opt.fit( D_tr, args.L1, args.L2, args.L, D_te, validation = 'mce' )
    elif args.validation.lower( ) == 'loo':
        opt.fit( D_tr, args.L1, args.L2, args.L, D_te, validation = 'loo' )
    elif args.validation.lower( )[ : 5 ] == 'kfold':
        K = int( args.validation.lower( )[ 5 : ] )
        opt.fit( D_tr, args.L1, args.L2, args.L, D_te, validation = 'kfold', K = K )
    else:
        print( 'Invalid validation strategy.' )
        sys.exit( 1 )
    # end if

    print( 'Final model: ' + str( m.m_P ) )

    X_tr = D_tr[ : , : D_tr.shape[ 1 ] - 1 ]
    y_tr = numpy.asmatrix( D_tr[ : , -1 ] ).T
    X_te = D_te[ : , : D_te.shape[ 1 ] - 1 ]
    y_te = numpy.asmatrix( D_te[ : , -1 ] ).T

    y_pred = m( X_te, True )
    y_pred = numpy.where( y_pred >= 1, 1, 0)
    y_true = numpy.where( y_te >= 1, 1, 0).ravel()

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()


    # X_tr = D_tr[ : , : D_tr.shape[ 1 ] - 1 ]
    # y_tr = numpy.asmatrix( D_tr[ : , -1 ] ).T
    # X_te = D_te[ : , : D_te.shape[ 1 ] - 1 ]
    # y_te = numpy.asmatrix( D_te[ : , -1 ] ).T
    # model = SGDClassifier(
    #     loss='hinge',            # SVM loss
    #     penalty='elasticnet',    # Elastic Net regularization
    #     alpha=2,            # Regularization strength (like 1/C)
    #     l1_ratio=0.5,            # Mix between L1 (1.0) and L2 (0.0)
    #     max_iter=args.epochs
    # ) 

    # model.fit( numpy.asarray( X_tr ), numpy.asarray( y_tr ).ravel() )
    # # Predict and evaluate
    # y_pred = model.predict( numpy.asarray( X_te ) )
    # y_pred = numpy.where( y_pred >= 1, 1, 0 )
    # y_te_adj = numpy.where( y_te >= 1, 1, 0 )

    # K_te = Confussion( model, X_te, y_te )
    # sensibility = K_te[ 1 ]
    # specificity = K_te[ 2 ]
    # accuracy = K_te[ 3 ]
    # F1 = K_te[ 4 ]
    # print( 'SGDClassifier result' )
    # print( 'Sensibility: ', sensibility )
    # print( "Specificity: ", specificity )
    # print( "Accuracy: ", accuracy )
    # print( "F1: ", F1 )

    # K_te = Confussion( m, X_te, y_te )
    # sensibility = K_te[ 1 ]
    # specificity = K_te[ 2 ]
    # accuracy = K_te[ 3 ]
    # F1 = K_te[ 4 ]
    # print( 'SVM result' )
    # print( 'Sensibility: ', sensibility )
    # print( "Specificity: ", specificity )
    # print( "Accuracy: ", accuracy )
    # print( "F1: ", F1 )

    # print(model.coef_)

# end if
# eof - main.py