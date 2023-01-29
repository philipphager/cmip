# CMIP - Conditional Mutual Information with the logging Policy
CMIP implementation for the paper: `An Offline Metric for the Debiasedness of Click Models`, currently under review at SIGIR 2023.

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
from cmip import CMIP

metric = CMIP()
metric(y_predict, y_logging_policy, y_true, n)
> 0.2687 # The policy predicting y_predict is not debiased w.r.t. the logging policy.
```
## Installation
The package will be made available on [pypi](https://pypi.org/) on acceptance.
