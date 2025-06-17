import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class BootstrapGaussianNB(BaseEstimator, ClassifierMixin):
    def __init__(self, n_bootstrap=100, assume_uniform_prior=True):
        self.n_bootstrap = n_bootstrap
        self.assume_uniform_prior = assume_uniform_prior

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        means = np.full((n_classes, n_features), np.nan)
        vars_ = np.full((n_classes, n_features), np.nan)

        for idx, cls in enumerate(self.classes_):
            X_cls = X[y == cls]
            boot_means = []
            boot_vars = []

            for _ in range(self.n_bootstrap):
                sample = X_cls[np.random.choice(X_cls.shape[0], X_cls.shape[0], replace=True)]
                boot_means.append(np.nanmean(sample, axis=0))
                boot_vars.append(np.nanvar(sample, axis=0) + 1e-9)  # Add small value for stability

            means[idx] = np.nanmean(boot_means, axis=0)
            vars_[idx] = np.nanmean(boot_vars, axis=0)

        self.theta_ = means
        self.sigma_ = vars_

        if self.assume_uniform_prior:
            self.class_prior_ = np.ones(n_classes) / n_classes
        else:
            class_counts = np.array([np.sum(y == c) for c in self.classes_])
            self.class_prior_ = class_counts / class_counts.sum()

        return self

    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_probs = np.full((n_samples, n_classes), -np.inf)

        for idx, cls in enumerate(self.classes_):
            mean = self.theta_[idx]
            var = self.sigma_[idx]

            valid = ~np.isnan(X) & ~np.isnan(mean) & ~np.isnan(var)

            log_likelihood = np.full(n_samples, 0.0)
            for i in range(n_samples):
                vi = valid[i]
                if not np.any(vi):
                    log_likelihood[i] = -np.inf
                    continue
                xi = X[i, vi]
                mi = mean[vi]
                si = var[vi]
                log_likelihood[i] = -0.5 * np.sum(np.log(2 * np.pi * si) + ((xi - mi) ** 2) / si)

            log_prior = np.log(self.class_prior_[idx])
            log_probs[:, idx] = log_likelihood + log_prior

        # Numerical stability
        log_probs -= np.nanmax(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs_sum = np.nansum(probs, axis=1, keepdims=True)
        probs_sum[probs_sum == 0] = np.nan  # Avoid division by zero
        probs /= probs_sum

        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.nanargmax(probs, axis=1)]
