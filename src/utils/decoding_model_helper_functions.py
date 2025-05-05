import numpy as np
from src.utils import list_manipulation_functions as lmf

def kfold_split(y, kfolds):
    if kfolds > len(y):
        raise Exception('kfolds must be equal or lower than target size')
    class_vector, class_counts = np.unique(y, return_counts=True)
    folds_samples = [[]] * kfolds

    for class_vals in class_vector:
        I_y_aux = np.where(y == class_vals)[0]

        fold_size = np.floor(I_y_aux.shape[0] / kfolds).astype(int)

        for kk in range(kfolds):
            samples = np.random.choice(I_y_aux, fold_size, replace=False)
            folds_samples[kk] = np.append(folds_samples[kk], samples).astype(int)
            I_y_aux = np.delete(I_y_aux, np.in1d(I_y_aux, samples))

        if len(I_y_aux) > 0:
            current_folds_sizes = np.array(list(map(len, folds_samples)))
            I_sorted_arg_folds = np.argsort(current_folds_sizes + np.random.rand(*current_folds_sizes.shape))
            remaining_folds = np.random.choice(kfolds, len(I_y_aux), replace=False)
            counter = 0
            remaining_folds = I_sorted_arg_folds[0:I_y_aux.shape[0]]
            for kk in remaining_folds:
                folds_samples[kk] = np.append(folds_samples[kk], I_y_aux[counter]).astype(int)
                counter += 1

    return folds_samples


def kfold_split_continuous(y, kfolds):

    y_idx = np.arange(y.shape[0])
    folds_samples = [[]] * kfolds
    fold_size = np.floor(y_idx.shape[0] / kfolds).astype(int)
    start_idx = np.arange(0,y_idx.shape[0],fold_size)
    
    for kk in range(kfolds):
        if kk < kfolds-1:
            samples = np.arange(start_idx[kk],start_idx[kk]+fold_size)
            folds_samples[kk] = np.append(folds_samples[kk], samples).astype(int)
        else:
            samples = np.arange(start_idx[kk],y.shape[0])
            folds_samples[kk] = np.append(folds_samples[kk], samples).astype(int)


    return folds_samples


def get_fold_trials(X,y,fold,num_of_folds):

    window_size = int(np.floor(X.shape[0]/num_of_folds))
    start_idx = np.arange(0,X.shape[0],window_size)

    if fold==(num_of_folds):
        trials_testing_set =  np.arange(start_idx[fold-1],X.shape[0]).astype(int)
    elif (fold == 0) | (fold > num_of_folds):
        raise ValueError('Fold number must be a integer between 1 and num_of_folds') 
    else:
        trials_testing_set =  np.arange(start_idx[fold-1],start_idx[fold]).astype(int)

    trials_training_set = np.setdiff1d(range(X.shape[0]),trials_testing_set)

    X_train = X[trials_training_set,:].copy()
    y_train = y[trials_training_set].copy()

    X_test = X[trials_testing_set,:].copy()
    y_test = y[trials_testing_set].copy()

    return X_train,y_train,X_test,y_test,trials_training_set,trials_testing_set


    
def kfold_run(X, y, folds_samples, test_fold):
    
    training_folds = np.delete(np.arange(len(folds_samples)), test_fold)
    # training_folds = np.setdiff1d(np.arange(len(folds_samples)), test_fold)

    idx_train = lmf.select(folds_samples, training_folds,concatenate=True)
    idx_test = folds_samples[test_fold]

    X_train = X[idx_train, :].copy()
    X_test = X[idx_test, :].copy()

    y_train = y[idx_train].copy()
    y_test = y[idx_test].copy()

    return X_train, X_test, y_train, y_test
    
def resampling(X, y,method = 'original'):
    
    classes, counts = np.unique(y, return_counts=True)
    
    if method == 'min':
        num_of_resamples = np.nanmin(counts)
    elif method == 'original':
        y_resampled = y.copy()
        X_resampled = X.copy()
        return X_resampled, y_resampled
    else:
        raise Exception('Method not defined')
    
    y_resampled = []
    X_resampled = []
    
    for cla in classes:
        I_to_resample = np.where(y == cla)[0]
        I_resampled = np.random.choice(I_to_resample, num_of_resamples, replace=False)
        y_resampled.append(y[I_resampled])
        X_resampled.append(X[I_resampled, :])
    X_resampled = np.concatenate(X_resampled,0)
    y_resampled = np.concatenate(y_resampled)
    return X_resampled, y_resampled


def resampling_old(X, y,method = 'original'):
    
    classes, counts = np.unique(y, return_counts=True)
    
    if method == 'max':
        num_of_resamples = np.nanmax(counts)
    elif method == 'min':
        num_of_resamples = np.nanmin(counts)
    elif method == 'original':
        y_resampled = y.copy()
        X_resampled = X.copy()
        return X_resampled, y_resampled
    else:
        raise Exception('Method not defined')
    
    y_resampled = []
    X_resampled = []
    
    for cla in classes:
        I_to_resample = np.where(y == cla)[0]
        I_resampled = np.random.choice(I_to_resample, num_of_resamples, replace=True)
        y_resampled.append(y[I_resampled])
        X_resampled.append(X[I_resampled, :])
    X_resampled = np.concatenate(X_resampled,0)
    y_resampled = np.concatenate(y_resampled)

    return X_resampled, y_resampled


def leave_one_out_split(X, y):
    """
    Perform leave-one-out cross-validation (LOOCV) split.

    Parameters:
    - X: Input features (array-like or dataframe).
    - y: Target variable (array-like).

    Returns:
    - List of tuples containing (X_train, X_test, y_train, y_test) for each split.
    """
    loo_splits = []
    num_samples = len(X)
    num_samples = X.shape[0]
    for i in range(num_samples):
        # Create train-test split for the current sample
        X_train_idx = np.delete(X, i, axis=0)
        X_test_idx = X[i:i+1]
        y_train_idx = np.delete(y, i, axis=0)
        y_test_idx = y[i:i+1]

        loo_splits.append((X_train_idx, X_test_idx, y_train_idx, y_test_idx))

    return loo_splits


def leave_two_out_split(X, y):
    """
    Perform leave-two-out cross-validation split, always taking one from each group.

    Parameters:
    - X: Input features (array-like or dataframe).
    - y: Target variable (array-like).

    Returns:
    - List of tuples containing (X_train, X_test, y_train, y_test) for each split.
    """
    # Get unique classes
    classes = np.unique(y)
    num_classes = len(classes)
    
    # Get the number of samples per class
    samples_per_class = [np.sum(y == cls) for cls in classes]
    
    # Check if all classes have the same number of samples
    if len(set(samples_per_class)) != 1:
        raise ValueError("All classes must have the same number of samples for this implementation.")
    
    num_samples_per_class = samples_per_class[0]
    
    lto_splits = []
    
    for i in range(num_samples_per_class):
        # Indices to leave out (one from each class)
        leave_out_indices = [np.where(y == cls)[0][i] for cls in classes]
        
        # Create train-test split for the current leave-out indices
        X_train_idx = np.delete(X, leave_out_indices, axis=0)
        X_test_idx = X[leave_out_indices]
        y_train_idx = np.delete(y, leave_out_indices, axis=0)
        y_test_idx = y[leave_out_indices]
        
        lto_splits.append((X_train_idx, X_test_idx, y_train_idx, y_test_idx))
    
    return lto_splits


def train_test_split(X, y, test_size=0.25, random_state=None):
    """Split the data into train and test sets.

    Parameters:
    X : array-like of shape (n_samples, n_features, n_points)
        Input data.
    y : array-like of shape (n_samples,)
        Target values.
    test_size : float, default=0.25
        Proportion of the data to include in the test split.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Returns:
    x_train : array-like of shape (n_train_samples, n_features, n_points)
        Training data.
    x_test : array-like of shape (n_test_samples, n_features, n_points)
        Test data.
    y_train : array-like of shape (n_train_samples,)
        Training labels.
    y_test : array-like of shape (n_test_samples,)
        Test labels.
    """
    # Setting random seed if provided
    if random_state is not None:
        np.random.seed(random_state)
    
    # Getting the number of samples
    n_samples = X.shape[0]
    
    # Determining the number of samples for the test set
    n_test_samples = int(test_size * n_samples)
    
    # Randomly selecting indices for the test set
    test_indices = np.random.choice(n_samples, size=n_test_samples, replace=False)
        
    # Creating masks for train and test indices
    train_mask = np.ones(n_samples, dtype=bool)
    train_mask[test_indices] = False
    test_mask = ~train_mask
    
    # Creating train and test sets
    x_train, x_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    return x_train, x_test, y_train, y_test

    