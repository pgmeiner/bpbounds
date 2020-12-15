# bpbounds

Implementation of the nonparametric bounds for the average treatment effect (ATE) or average causal effect (ACE) of [Balke and Pearl (1997)](https://doi.org/10.1080/01621459.1997.10474074). This is a python version of the R-package `bpbounds`.

## Installation
To install the `bpbounds` package from github execute the following commands in your terminal:

```setup
pip install git+https://github.com/pgmeiner/bpbounds.git
```

```setup
pip install -r requirements.txt
```

## Usage

The formulas for the Balke-Pearl bounds apply in situations were three random variables `X`, `Y`, and `Z` are related via an instrumental setting. That means
`Z->X->Y` and we can have confounding variables `C` for `X` and `Y`. To be precise We assume the following situation:
* `Z` is a instrumental variable (i.e. `Z` and `C` are independent, `Z` and `X` are dependent, and `Z`,`Y` are independent when conditioned on `{X,C}`)
* `X` is a cause of `Y` (i.e. `X->Y`)
* `C` is a unobserved confounder (i.e. `X<-C->Y`)

The Balke-Pearl bounds give lower and upper bounds for the average causal effect between `X` and `Y` when `X`, `Y` are binary and `Z` is binary or ternary:
```
bplb <= (P(Y|do(X=1)) - P(Y|do(X=0))) <= bpub
```
It has to be checked before if `Z` is a instrumental variable. However, the package also calculates the so-called `instrumental inequality`
that is a necessary condition that `Z` is a instrumental variable and can serve as a test to reject inadequate situations 
(But such a test does not replace non data-driven arguments that `Z` is a instrumental variable).
When we have additionally a monotonic situation we can sometimes get stronger bounds. Also this is an assumption that can be checked via
an inequality but since this is also just a necessary condition it is not sufficient for showing that the relation between
`X` and `Y` is monotonic (see references for more details).

The main function `ace_balke_pearl_bounds` expects pandas series as parameters that contain data from `X`, `Y`, and `Z`. 
It calculates all bounds as well as all inequalities and returns a dictionary that can be printed out by using `print_results`.

```python
import pandas as pd
from bpbounds.balke_pearl_bounds import ace_balke_pearl_bounds, print_results

x = [1, 0, 1, 0, 1]
y = [1, 1, 0, 0, 1]
z = [1, 0, 1, 0, 1]
df = pd.DataFrame({'x': x, 'y': y, 'z': z})
res = ace_balke_pearl_bounds(df['x'], df['y'], df['z'])

print_results(res)
```
Output:
```
Instrumental inequality: True
Causal parameter Lower bound Upper bound
            ACE  0.1666667 0.1666667
   P(Y|do(X=0))  0.5000000 0.5000000
   P(Y|do(X=1))  0.6666667 0.6666667
            CRR  1.3333333 1.3333333
Monotonicity inequality: True
Causal parameter Lower bound Upper bound
            ACE  0.1666667 0.1666667
   P(Y|do(X=0))  0.5000000 0.5000000
   P(Y|do(X=1))  0.6666667 0.6666667
            CRR  1.3333333 1.3333333
```

The results dictionary `res` contains the following entries:

* `nzcats`: number of categories in `z`
* `inequality`: True if instrumental inequality is fulfilled. If False then we should not trust the lower and upper bounds at all (since we are not in an instrumental setting).
* `bplb`: lower bound for ACE
* `bpup`: upper bound for ACE
* `p11low`: lower bound for `P(Y|do(X=1))`
* `p11upp`: upper bound for `P(Y|do(X=1))`
* `p10low`: lower bound for `P(Y|do(X=0))`
* `p10upp`: upper bound for `P(Y|do(X=0))`
* `crrlb`: lower bound for Causal Risk Ratio (CRR) := `P(Y=1|do(X=1)) / P(Y=1|do(X=0))`
* `crrub`: upper bound for Causal Risk Ratio
* `monoinequality`: True if monotonicity inequality is fulfilled. If False we are not in a monotonic situation and the following entries are `0
* `monobplb`: monotonic lower bound for ACE
* `monobpub`: monotonic upper bound for ACE
* `monop11low`: monotonic lower bound for `P(Y|do(X=1))`
* `monop11upp`: monotonic upper bound for `P(Y|do(X=1))`
* `monop10low`: monotonic lower bound for `P(Y|do(X=0))`
* `monop10upp`: monotonic upper bound for `P(Y|do(X=0))`
* `monocrrlb`: monotonic lower bound for Causal Risk Ratio
* `monocrrub`: monotonic upper bound for Causal Risk Ratio

## Authors
Peter Gmeiner (maintainer, peter.gmeiner@algobalance.com).

## References

Balke A, Pearl J. Bounds on Treatment Effects from studies with imperfect compliance. Journal of the American Statistical Association, 1997, 92, 439, 1171-1176, doi: [10.1080/01621459.1997.10474074](https://doi.org/10.1080/01621459.1997.10474074).
