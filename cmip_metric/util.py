import numpy as np
import numpy.typing as npt


def padding_mask(n: npt.NDArray[np.int_], n_batch: int, n_results: int) -> npt.NDArray:
    mask = np.tile(np.arange(n_results), (n_batch, 1))
    return mask < n.reshape(-1, 1)


def add_label(x: npt.NDArray, label: int) -> npt.NDArray:
    """
    Adds a new column to a matrix containing a given label
    """
    labels = np.full((len(x), 1), label)
    return np.hstack([x, labels])


def hstack(
    y_predict: npt.NDArray[np.float64],
    y_logging_policy: npt.NDArray[np.float64],
    y_true: npt.NDArray[np.float64],
) -> npt.NDArray:
    """
    Flatten all three arrays to 1d and stack them into a matrix of size:
    (n_results * n_queries) x 3
    Each row contains thus one observation of:
    (y_predict, y_logging_policy, y_true) or (x, y, z)
    """
    return np.column_stack(
        [
            y_predict.ravel(),
            y_logging_policy.ravel(),
            y_true.ravel(),
        ]
    )
