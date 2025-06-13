import warnings
from typing import Dict, Optional, Union
import numpy as np
from sklearn.preprocessing import StandardScaler

class ManifoldLearner:
    """
    Enhanced manifold learning with additional techniques:
    
    Linear Methods:
    - PCA, KernelPCA, SparsePCA, FactorAnalysis, FastICA, DictionaryLearning, LDA
    
    Nonlinear Manifold Learning:
    - UMAP, PaCMAP, TRIMAP, t-SNE, Isomap, LLE variants, Spectral, MDS, PHATE
    
    Usage:
    >>> ml = ManifoldLearner()
    >>> # Get all embeddings
    >>> all_embeds = ml.run_all_methods(X, n_components=2)
    >>> # Get specific method
    >>> phate_embed = ml.fit_transform(X, method='phate')
    """

    def __init__(self, scale_data: bool = False):
        """
        Parameters:
        -----------
        scale_data : bool (default=True)
            Whether to standard-scale data before processing (recommended for most methods)
        """
        self.scale_data = scale_data
        self.scaler = StandardScaler() if scale_data else None
        self.embedding_ = None
        self.model = None
        self._imported_modules = {}  # Cache for imported modules

    def _import_sklearn_submodule(self, submodule: str, class_name: str):
        """Lazy import for sklearn submodules"""
        if submodule not in self._imported_modules:
            try:
                module = __import__(f'sklearn.{submodule}', fromlist=[class_name])
                self._imported_modules[submodule] = module
            except ImportError as e:
                warnings.warn(f"scikit-learn {submodule} submodule not available: {str(e)}")
                raise
        return getattr(self._imported_modules[submodule], class_name)

    def _try_import(self, package_name: str, class_name: Optional[str] = None):
        """Attempt to import a package or specific class with warning on failure"""
        try:
            module = __import__(package_name)
            if class_name:
                return getattr(module, class_name)
            return module
        except ImportError:
            warnings.warn(f"Package {package_name} not available. Some functionality may be limited.")
            return None

    def _prepare_data(self, X):
        """Apply scaling if enabled"""
        if self.scale_data:
            return self.scaler.fit_transform(X) if not hasattr(self.scaler, 'mean_') else self.scaler.transform(X)
        return X

    def fit_transform(self, X, method='umap', n_components=2, **kwargs):
        """
        Fit and transform data using specified manifold learning method.
        
        Parameters:
        -----------
        X : array-like
            Input data (n_samples, n_features)
        method : str
            Name of the manifold learning method
        n_components : int
            Output dimensionality
        **kwargs : dict
            Method-specific parameters
            
        Returns:
        --------
        Embedding array (n_samples, n_components)
        """
        X = self._prepare_data(X)
        method = method.lower()

        # Linear methods
        if method == 'pca':
            PCA = self._import_sklearn_submodule('decomposition', 'PCA')
            self.model = PCA(n_components=n_components, **kwargs)
        elif method == 'kpca':
            KernelPCA = self._import_sklearn_submodule('decomposition', 'KernelPCA')
            self.model = KernelPCA(n_components=n_components, **kwargs)
        elif method == 'sparsepca':
            SparsePCA = self._import_sklearn_submodule('decomposition', 'SparsePCA')
            self.model = SparsePCA(n_components=n_components, **kwargs)
        elif method == 'factor':
            FactorAnalysis = self._import_sklearn_submodule('decomposition', 'FactorAnalysis')
            self.model = FactorAnalysis(n_components=n_components, **kwargs)
        elif method == 'ica':
            FastICA = self._import_sklearn_submodule('decomposition', 'FastICA')
            self.model = FastICA(n_components=n_components, **kwargs)
        elif method == 'dictlearn':
            DictionaryLearning = self._import_sklearn_submodule('decomposition', 'DictionaryLearning')
            self.model = DictionaryLearning(n_components=n_components, **kwargs)
        elif method == 'lda':
            if 'y' not in kwargs:
                raise ValueError("LDA requires y labels")
            LinearDiscriminantAnalysis = self._import_sklearn_submodule('discriminant_analysis', 'LinearDiscriminantAnalysis')
            self.model = LinearDiscriminantAnalysis(n_components=n_components)
            self.model.fit(X, kwargs['y'])
            self.embedding_ = self.model.transform(X)
            return self.embedding_
        
        # Nonlinear methods
        elif method == 'umap':
            umap_module = self._try_import('umap')
            if umap_module is None:
                raise ImportError("UMAP package not installed. Install with: pip install umap-learn")
            self.model = umap_module.UMAP(n_components=n_components, **kwargs)
        elif method == 'pacmap':
            pacmap_module = self._try_import('pacmap')
            if pacmap_module is None:
                raise ImportError("PaCMAP package not installed. Install with: pip install pacmap")
            self.model = pacmap_module.PaCMAP(n_components=n_components, **kwargs)
        elif method == 'trimap':
            trimap_module = self._try_import('trimap')
            if trimap_module is None:
                raise ImportError("TRIMAP package not installed. Install with: pip install trimap")
            self.model = trimap_module.TRIMAP(n_components=n_components, **kwargs)
        elif method == 'tsne':
            TSNE = self._import_sklearn_submodule('manifold', 'TSNE')
            self.model = TSNE(n_components=n_components, **kwargs)
        elif method == 'mds':
            MDS = self._import_sklearn_submodule('manifold', 'MDS')
            self.model = MDS(n_components=n_components, **kwargs)
        elif method == 'isomap':
            Isomap = self._import_sklearn_submodule('manifold', 'Isomap')
            self.model = Isomap(n_components=n_components, **kwargs)
        elif method == 'lle':
            LocallyLinearEmbedding = self._import_sklearn_submodule('manifold', 'LocallyLinearEmbedding')
            self.model = LocallyLinearEmbedding(n_components=n_components, 
                                             method='standard', **kwargs)
        elif method == 'ltsa':
            LocallyLinearEmbedding = self._import_sklearn_submodule('manifold', 'LocallyLinearEmbedding')
            self.model = LocallyLinearEmbedding(n_components=n_components,
                                             method='ltsa', **kwargs)
        elif method == 'hessian':
            LocallyLinearEmbedding = self._import_sklearn_submodule('manifold', 'LocallyLinearEmbedding')
            self.model = LocallyLinearEmbedding(n_components=n_components,
                                             method='hessian', **kwargs)
        elif method == 'modified':
            LocallyLinearEmbedding = self._import_sklearn_submodule('manifold', 'LocallyLinearEmbedding')
            self.model = LocallyLinearEmbedding(n_components=n_components,
                                             method='modified', **kwargs)
        elif method == 'spectral':
            SpectralEmbedding = self._import_sklearn_submodule('manifold', 'SpectralEmbedding')
            self.model = SpectralEmbedding(n_components=n_components, **kwargs)
        elif method == 'phate':
            phate_module = self._try_import('phate')
            if phate_module is None:
                raise ImportError("PHATE package not installed. Install with: pip install phate")
            self.model = phate_module.PHATE(n_components=n_components, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}. See docstring for available methods.")
        
        self.embedding_ = self.model.fit_transform(X)
        return self.embedding_

    def run_all_methods(self, X, n_components=2, skip_methods: Optional[list] = None, 
                    method_params: Optional[Dict[str, Dict]] = None, **kwargs) -> Dict[str, np.ndarray]:
        """
        Run all available methods with intelligent grouping to avoid redundant computations.
        
        Parameters:
        -----------
        X : array-like
            Input data (n_samples, n_features)
        n_components : int
            Output dimensionality
        skip_methods : list, optional
            Methods to skip (e.g., ['lda'] if no labels available)
        method_params : dict, optional
            Dictionary of method-specific parameters where keys are method names and values
            are dictionaries of parameters for that method.
            Example: {'umap': {'n_neighbors': 15}, 'tsne': {'perplexity': 30}}
        **kwargs : dict
            Common parameters that will be passed to all methods
            
        Returns:
        --------
        Dict with method names as keys and embeddings as values


        Usage:
        # Skipping Methods and Common Parameters
        results = ml.run_all_methods(
            X, 
            n_components=2,
            skip_methods=['lda', 'dictlearn'],  # Skip these methods
            random_state=42  # Common parameter for all methods
        )

        # Method-Specific Parameters
        results = ml.run_all_methods(
            X,
            n_components=2,
            method_params={
                'umap': {'n_neighbors': 15, 'min_dist': 0.1},
                'tsne': {'perplexity': 30, 'early_exaggeration': 12},
                'phate': {'knn': 5, 'gamma': 0.5},
                'lle': {'n_neighbors': 20}
            },
            random_state=42  # Still applies to all methods
        )

        # Mixed Common and Method-Specific Parameters

        results = ml.run_all_methods(
            X,
            n_components=3,
            method_params={
                'umap': {'n_neighbors': 10},  # Overrides common n_neighbors for UMAP only
                'pacmap': {'n_neighbors': 15}
            },
            n_neighbors=5,  # Common parameter (used by methods that accept it)
            skip_methods=['lda', 'ica']
        )

        """
        X = self._prepare_data(X)
        skip_methods = skip_methods or []
        method_params = method_params or {}
        
        method_groups = {
            'linear': ['pca', 'kpca', 'sparsepca', 'factor', 'ica', 'dictlearn'],
            'nonlinear': ['umap', 'pacmap', 'trimap', 'tsne', 'mds', 'isomap',
                        'lle', 'ltsa', 'hessian', 'modified', 'spectral', 'phate']
        }
        
        embeddings = {}
        
        for group, methods in method_groups.items():
            for method in methods:
                if method in skip_methods:
                    continue
                    
                try:
                    # Combine common kwargs with method-specific params
                    params = kwargs.copy()  # Start with common params
                    params.update(method_params.get(method, {}))  # Add method-specific params
                    
                    if method == 'lda' and 'y' not in params:
                        continue
                        
                    print(f"Computing {method} with params: {params}...")
                    embeddings[method] = self.fit_transform(
                        X, 
                        method=method,
                        n_components=n_components,
                        **params
                    )
                except Exception as e:
                    print(f"Failed {method}: {str(e)}")
                    embeddings[method] = None
        
        return embeddings