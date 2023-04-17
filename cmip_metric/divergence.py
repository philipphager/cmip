from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from cmip_metric.util import add_label


class KLDivergence(ABC):
    @abstractmethod
    def __call__(self, p: npt.NDArray, q: npt.NDArray) -> float:
        pass


class ClassifierKLDivergence(KLDivergence):
    """
    Classifier-based estimator of the Kullback–Leibler divergence as in Eq. 3 and Alg. 1
    of [Mukherjee et al. 2019](http://auai.org/uai2019/proceedings/papers/403.pdf).
    """

    def __init__(
        self,
        classifier: Any = LGBMClassifier(),
        n_bootstrap: int = 5,
        eta: float = 1e-8,
    ):
        self.classifier = classifier
        self.n_bootstrap = n_bootstrap
        self.eta = eta

    def __call__(self, p: npt.NDArray, q: npt.NDArray) -> float:
        p = add_label(p, 1)
        q = add_label(q, 0)
        kl_divergences = [self.kl_divergence(p, q) for _ in range(self.n_bootstrap)]

        return np.mean(kl_divergences).item()

    def kl_divergence(self, p: npt.NDArray, q: npt.NDArray):
        p_train, p_test = train_test_split(p, test_size=0.5)
        q_train, q_test = train_test_split(q, test_size=0.5)

        train = np.vstack([p_train, q_train])
        classifier = self.train(self.classifier, train)

        p_predict = self.predict(classifier, p_test, self.eta)
        q_predict = self.predict(classifier, q_test, self.eta)

        # Point-wise likelihood ratio as in section 3.1
        p_ratio = p_predict / (1 - p_predict)
        q_ratio = q_predict / (1 - q_predict)

        # Estimate KL divergence using Donsker-Varadhan formulation
        return (np.log(p_ratio) - np.log(q_ratio.mean())).mean()

    @staticmethod
    def train(model: Any, train: npt.NDArray):
        x = train[:, :3]
        y = train[:, 3]
        model.fit(x, y)

        return model

    @staticmethod
    def predict(model: Any, test: npt.NDArray, eta: float):
        x = test[:, :3]
        y_predict = model.predict_proba(x)[:, 1]
        # Clip predictions to avoid exploding likelihood ratios
        return y_predict.clip(eta, 1 - eta)
