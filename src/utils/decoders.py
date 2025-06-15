import warnings
from typing import Dict, Optional, Union, List
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from src.utils import decoding_model_helper_functions as decoder_helper

class DecoderLearner:
    """
    Versatile decoding framework using various classifiers.

    Usage:
    >>> dl = DecoderLearner(scale_data=True)
    >>> acc = dl.run_classifier(X, y, decoder='random_forest', classifier_params={'n_estimators': 200})
    >>> results = dl.run_all_classifiers(X, y, skip=['qda'], classifier_params={'svc': {'kernel': 'linear'}})
    """

    def __init__(self, scale_data: bool = True):
        self.scale_data = scale_data
        self.scaler = StandardScaler() if scale_data else None
        self.model = None
        self._imported_modules = {}

    def _prepare_data(self, X):
        return self.scaler.fit_transform(X) if self.scale_data else X

    def _import_sklearn_classifier(self, submodule: str, class_name: str):
        """Lazy import for sklearn classifier"""
        if submodule not in self._imported_modules:
            try:
                module = __import__(f'sklearn.{submodule}', fromlist=[class_name])
                self._imported_modules[submodule] = module
            except ImportError as e:
                warnings.warn(f"scikit-learn {submodule} submodule not available: {str(e)}")
                raise
        return getattr(self._imported_modules[submodule], class_name)

    def _try_import_external(self, package_name: str, class_name: Optional[str] = None):
        try:
            module = __import__(package_name)
            return getattr(module, class_name) if class_name else module
        except ImportError:
            warnings.warn(f"Package {package_name} not available.")
            return None

    def _get_decoder_model(self, decoder: str, **kwargs):
        if decoder == 'gaussian_nb':
            GaussianNB = self._import_sklearn_classifier('naive_bayes', 'GaussianNB')
            return GaussianNB(**kwargs)
        
        elif decoder == 'bernoulli_nb':
            BernoulliNB = self._import_sklearn_classifier('naive_bayes', 'BernoulliNB')
            return BernoulliNB(**kwargs)
        
        elif decoder == 'multinomial_nb':
            MultinomialNB = self._import_sklearn_classifier('naive_bayes', 'MultinomialNB')
            return MultinomialNB(**kwargs)
        
        elif decoder == 'complement_nb':
            ComplementNB = self._import_sklearn_classifier('naive_bayes', 'ComplementNB')
            return ComplementNB(**kwargs)
        
        elif decoder == 'categorical_nb':
            CategoricalNB = self._import_sklearn_classifier('naive_bayes', 'CategoricalNB')
            return CategoricalNB(**kwargs)
        
        elif decoder == 'svc':
            SVC = self._import_sklearn_classifier('svm', 'SVC')
            return SVC(**kwargs)
        
        elif decoder == 'logreg':
            LogisticRegression = self._import_sklearn_classifier('linear_model', 'LogisticRegression')
            return LogisticRegression(**kwargs)
        
        elif decoder == 'random_forest':
            RF = self._import_sklearn_classifier('ensemble', 'RandomForestClassifier')
            return RF(**kwargs)
        
        elif decoder == 'qda':
            QDA = self._import_sklearn_classifier('discriminant_analysis', 'QuadraticDiscriminantAnalysis')
            return QDA(**kwargs)
        
        elif decoder == 'lda':
            LDA = self._import_sklearn_classifier('discriminant_analysis', 'LinearDiscriminantAnalysis')
            return LDA(**kwargs)
        
        elif decoder == 'xgboost':
            xgb = self._try_import_external('xgboost', 'XGBClassifier')
            if xgb is None:
                raise ImportError("XGBoost not installed. Install with: pip install xgboost")
            return xgb(**kwargs)
        
        else:
            raise ValueError(f"Unknown decoder: {decoder}")


    def run_classifier(self, X, y, kfolds=3, decoder='gaussian_nb',
                   classifier_params: Optional[Dict[str, Dict]] = None, **kwargs):
        """
        Run a single decoder with K-fold cross-validation.
        Supports both global kwargs and per-decoder classifier_params.
        """
        X = self._prepare_data(X)

        # Step 1: Shuffle label mapping
        unique_labels = np.unique(y)
        shuffled_labels = np.random.permutation(unique_labels)
        label_mapping = dict(zip(unique_labels, shuffled_labels))
        inverse_label_mapping = {v: k for k, v in label_mapping.items()}
        y_mapped = np.vectorize(label_mapping.get)(y)

        nb_decoders = ['gaussian_nb','bernoulli_nb','multinomial_nb','complement_nb','categorical_nb']

        # Step 2: Prepare k-fold samples
        folds_samples = decoder_helper.kfold_split_continuous(y_mapped, kfolds)
        y_pred = []

        # Step 3: Run K-fold cross-validation
        for test_fold in range(kfolds):
            X_train, X_test, y_train, y_test = decoder_helper.kfold_run(X, y_mapped, folds_samples, test_fold)

            # Start from scratch each fold
            decoder_kwargs = kwargs.copy()
            if classifier_params:
                decoder_kwargs.update(classifier_params)

            # Check for 'uniform' before modifying
            if decoder in nb_decoders and decoder_kwargs.get("priors") == "uniform":
                classes = np.unique(y_train)
                decoder_kwargs["priors"] = np.ones(len(classes)) / len(classes)


            model = self._get_decoder_model(decoder, **decoder_kwargs)
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)
            y_pred_fold = [model.classes_[self.random_argmax(p)] for p in probs]
            y_pred.append(y_pred_fold)

        y_pred = np.concatenate(y_pred)
        y_pred_restored = np.vectorize(inverse_label_mapping.get)(y_pred)

        return y_pred_restored.astype(int)


    def run_all_classifiers(self, 
                            X, 
                            y, 
                            kfolds=3, 
                            skip: Optional[List[str]] = None,
                            classifier_params: Optional[Dict[str, Dict]] = None,
                            **kwargs) -> Dict[str, float]:
        """
        Run all supported classifiers.

        Parameters:
        - skip: list of classifier names to skip
        - classifier_params: dict of decoder-specific params
        - kwargs: common parameters passed to all decoders

        Returns:
        - Dictionary with decoder names and their cross-validation accuracy
        """
        classifiers = ['naive_bayes', 'svc', 'logreg', 'random_forest', 'qda', 'lda', 'xgboost']
        skip = skip or []
        classifier_params = classifier_params or {}

        results = {}
        for decoder in classifiers:
            if decoder in skip:
                continue
            try:
                params = kwargs.copy()
                params.update(classifier_params.get(decoder, {}))
                print(f"Running {decoder} with params: {params}")
                y_pred = self.run_classifier(X, y, kfolds=kfolds, decoder=decoder, classifier_params=params)
                accuracy = np.mean(y_pred == y)
                results[decoder] = accuracy
            except Exception as e:
                print(f"Failed {decoder}: {str(e)}")
                results[decoder] = None
        return results

    def random_argmax(self, probabilities, epsilon=np.finfo(float).eps):
        """
        Adds small random noise to the probabilities and returns the index of the maximum value.
        
        Parameters:
        probabilities (array-like): Array of class probabilities for a single sample.
        epsilon (float): Standard deviation of the Gaussian noise to add.
        
        Returns:
        int: Index of the selected class.
        """
        noise = np.random.normal(loc=0.0, scale=epsilon, size=probabilities.shape)
        noisy_probs = probabilities + noise
        return np.argmax(noisy_probs)


    ''' 
    def run_classifier(self, X, y, kfolds=3):
        
        from sklearn.naive_bayes import GaussianNB

        def random_argmax(probabilities):
            """
            Selects a class index randomly among those with the highest probability.
            
            Parameters:
            probabilities (array-like): Array of class probabilities for a single sample.
            
            Returns:
            int: Index of the selected class.
            """
            max_prob = np.nanmax(probabilities)
            candidates = np.flatnonzero(probabilities == max_prob)
            return np.random.choice(candidates)


        folds_samples = decoder_helper.kfold_split_continuous(y, kfolds)
        y_pred = []

        for test_fold in range(kfolds):
            X_train, X_test, y_train, y_test = decoder_helper.kfold_run(X, y, folds_samples, test_fold)

            priors_in = np.ones(np.unique(y_train).shape[0]) / np.unique(y_train).shape[0]

            gnb = GaussianNB(priors=priors_in)
            gnb.fit(X_train, y_train)
            predict_probability = gnb.predict_proba(X_test)

            y_pred_fold = []
            for probs in predict_probability:
                selected_class_index = random_argmax(probs)
                y_pred_fold.append(gnb.classes_[selected_class_index])

            y_pred.append(np.array(y_pred_fold).astype(int))

        y_pred = np.concatenate(y_pred)
        return y_pred
        '''



    '''

    def run_classifier(self, X, y, kfolds=3):

        
        # Step 1: Generate a random label mapping
        unique_labels = np.unique(y)
        shuffled_labels = np.random.permutation(unique_labels)
        label_mapping = dict(zip(unique_labels, shuffled_labels))
        inverse_label_mapping = {v: k for k, v in label_mapping.items()}

        # Step 2: Apply the mapping to y
        y_mapped = np.vectorize(label_mapping.get)(y)

        # Step 3: Proceed with training and prediction using y_mapped
        folds_samples = decoder_helper.kfold_split_continuous(y_mapped, kfolds)
        y_pred = []

        for test_fold in range(kfolds):
            X_train, X_test, y_train, y_test = decoder_helper.kfold_run(X, y_mapped, folds_samples, test_fold)

            unique_classes = np.unique(y_train)
            priors_in = np.ones(len(unique_classes)) / len(unique_classes)
            gnb = GaussianNB(priors=priors_in)
            gnb.fit(X_train, y_train)
            predict_probability = gnb.predict_proba(X_test)

            y_pred_fold = []
            for probs in predict_probability:
                selected_class_index = self.random_argmax(probs)
                y_pred_fold.append(gnb.classes_[selected_class_index])

            y_pred.append(y_pred_fold)

        # Concatenate predictions from all folds
        y_pred = np.concatenate(y_pred)

        # Step 4: Restore original labels in predictions
        y_pred_restored = np.vectorize(inverse_label_mapping.get)(y_pred)

        return y_pred_restored.astype(int)
    '''