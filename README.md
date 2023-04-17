# CMIP - Conditional Mutual Information with the logging Policy
CMIP implementation from the 2023 SIGIR paper: `An Offline Metric for the Debiasedness of Click Models`.

The metric quantifies the mutual information between a new click model policy and the production system that collected the train dataset (logging policy), conditional on human relevance judgments. CMIP quantifies the degree of debiasedness (see paper for details). A policy is said to be debiased w.r.t. its logging policy with a `cmip <= 0`.  

## Example
```Python
import numpy as np

n_queries = 1_000
n_results = 25

# Human relevance annotations per query-document pair
y_true = np.random.randint(5, size=(n_queries, n_results))
# Relevance scores of the logging policy
y_logging_policy = y_true + np.random.randn(n_queries, n_results)
# Relevance scores of a new policy (in this case, strongly dependent on logging policy) 
y_predict = y_logging_policy + np.random.randn(n_queries, n_results)
# Number of documents per query, used for masking
n = np.full(n_queries, n_results)
```

```Python
from cmip_metric import CMIP

metric = CMIP()
metric(y_predict, y_logging_policy, y_true, n)
> 0.2687 # The policy predicting y_predict is not debiased w.r.t. the logging policy.
```
## Installation
```
pip install cmip-metric
```

## Reference

**Note: To be published at:**

```
@inproceedings{Deffayet2023Debiasedness,
  author = {Romain Deffayet and Philipp Hager and Jean-Michel Renders and Maarten de Rijke},
  title = {An Offline Metric for the Debiasedness of Click Models},
  booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR`23)},
  organization = {ACM},
  year = {2023},
}
```

## License
This project uses the [MIT license](https://github.com/philipphager/CMIP/blob/main/LICENSE).
