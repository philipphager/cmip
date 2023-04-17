import numpy as np


def nearest_neighbor_bootstrap(split1: np.ndarray, split2: np.ndarray):
    """
    This method uses two dataset splits containing observations (x, y, z) to
    generate a third dataset in which x, y are independent conditioned on z.

    This is done by iterating over the first split u1, and finding a nearest
    neighbor in u2 based on the z values. Then we swap the y values of both entries.
    Given that z are the true relevance labels, we avoid the nearest neighbor search
    and from all entries of u2 with a given relevance label to replace the entries
    in u1. This method needs to be adjusted when moving away from int relevance
    labels between 0-4 as z / y_true.

    :param split1: A tensor containing a third of the original dataset
    :param split2: A tensor containing another third of the original dataset
    :return: Split1 with the y entries replaced with samples from split2
    """
    split1 = split1.copy()
    split2 = split2.copy()
    z1 = split1[:, 2]
    z2 = split2[:, 2]

    for z in np.unique(z2):
        idx1 = np.argwhere(z1 == z).ravel()
        idx2 = np.argwhere(z2 == z).ravel()

        sample_idx = np.random.randint(len(idx2), size=(len(idx1),))
        split1[idx1, 1] = split2[idx2[sample_idx], 1]

    return split1
