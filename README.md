# BatchBALD Redux
> Clean reimplementation of \"BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning\"


For an introduction & more information, see [https://blackhc.github.io/BatchBALD/](https://blackhc.github.io/BatchBALD/). The paper can be found at [http://arxiv.org/abs/1906.08158](http://arxiv.org/abs/1906.08158). The documentation for this version can be found under [https://github.com/BlackHC/batchbald_redux/](https://github.com/BlackHC/batchbald_redux/).

The original implementation used in the paper is available at [https://github.com/BlackHC/BatchBALD](https://github.com/BlackHC/BatchBALD).

We are grateful for fastai's [nbdev](https://nbdev.fast.ai/) which is powering this package.

For more information, explore the sections and notebooks in the left-hand menu.
The code is available on [https://github.com/BlackHC/batchbald_redux](https://github.com/BlackHC/batchbald_redux),
and the website on [https://blackhc.github.io/batchbald_redux](https://blackhc.github.io/batchbald_redux).

## Install

`pip install batchbald_redux`

## Motivation

BatchBALD is an algorithm and acquisition function for Active Learning in a Bayesian setting using BNNs and MC dropout.

The aquisition function is the mutual information between the joint of a candidate batch and the model parameters $\omega$:

{% raw %}
$$a_{\text{BatchBALD}}((y_b)_B) = I[(y_b)_B;\omega]$$
{% endraw %}

The best candidate batch is one that maximizes this acquisition function.

In the paper, we show that this function satisfies sub-modularity, which provides us an optimality guarantee for a greedy algorithm. The candidate batch is selected using greedy expansion.

Joint entropies are hard to estimate and, for everything to work, one also has to use consistent MC dropout, which keeps a set of dropout masks fixed while scoring the pool set.

To aid reproducibility and baseline reproduction, we provide this simpler and clearer reimplementation.

## Please cite us

```
@misc{kirsch2019batchbald,
    title={BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning},
    author={Andreas Kirsch and Joost van Amersfoort and Yarin Gal},
    year={2019},
    eprint={1906.08158},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## How to use

We provide a simple example experiment that uses this package [here](docs/example_experiment.html).

To get a candidate batch using BatchBALD, we provide a simple API in [`batchbald_redux.batchbald.py`](batchbald_redux/batchbald.py):


<h4 id="get_batchbald_batch" class="doc_header"><code>get_batchbald_batch</code><a href="https://github.com/blackhc/batchbald_redux/tree/master/batchbald_redux/batchbald.py#L67" class="source_link" style="float:right">[source]</a></h4>

> <code>get_batchbald_batch</code>(**`log_probs_N_K_C`**:`Tensor`, **`batch_size`**:`int`, **`num_samples`**:`int`, **`dtype`**=*`None`*, **`device`**=*`None`*)




We also provide a simple implementation of consistent MC dropout in [`batchbald_redux.consistent_mc_dropout.py`](batchbald_redux/consistent_mc_dropout.py).
