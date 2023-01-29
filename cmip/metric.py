import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split

from cmip.divergence import KLDivergence, ClassifierKLDivergence
from cmip.knn import nearest_neighbor_bootstrap
from cmip.util import hstack, padding_mask


class CMIP:
    """
    Conditional Mutual Information towards the logging Policy (CMIP).

    Implements mutual information estimator from:
    Classifier-based Conditional Mutual Information (CCMI) from
    [Mukherjee et al. 2019](http://auai.org/uai2019/proceedings/papers/403.pdf).

    Main idea follows the mimic and classify schema by [Sen et al., 2017].
    Given the joint distribution P(X, Y, Z), we mimic a dataset in which
    X and Y are independent given Z using the knn method in [Sen et al., 2017].

    The conditional mutual information is then the KL divergence between the original
    distribution and the marginal distribution in which X and Y are independent.
    """

    def __init__(
        self,
        kl_divergence: KLDivergence = ClassifierKLDivergence(),
        n_bootstrap: int = 5,
        random_state: int = 0,
    ):
        self.kl_divergence = kl_divergence
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

    def __call__(
        self,
        y_predict: npt.NDArray[np.float64],
        y_logging_policy: npt.NDArray[np.float64],
        y_true: npt.NDArray[np.float64],
        n: npt.NDArray[np.int_],
    ) -> float:
        n_batch, n_results = y_true.shape
        mask = padding_mask(n, n_batch, n_results)
        dataset = hstack(y_predict, y_logging_policy, y_true)
        dataset = dataset[mask.ravel()]
        scores = []

        for _ in range(self.n_bootstrap):
            split1, split2 = train_test_split(
                dataset,
                test_size=0.5,
                random_state=self.random_state,
            )
            split2 = nearest_neighbor_bootstrap(split1, split2)
            scores.append(self.kl_divergence(split1, split2))

        return np.mean(scores).item()
