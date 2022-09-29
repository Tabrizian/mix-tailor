# MixTailor: Mixed Gradient Aggregation for Robust Learning Against Tailored Attacks

Code for [MixTailor: Mixed Gradient Aggregation for Robust Learning Against Tailored Attacks](https://arxiv.org/pdf/2207.07941.pdf) to appear in TMLR.


## Dependencies

We recommend using Anaconda to install the following packages:

* Python 3.7.1
* PyTorch
* TensorboardX
* Pyyaml
* Scipy

```
conda install pytorch torchvision -c pytorch
pip install pyyaml scipy tensorboardx
```


## General Overview

The code mainly cosists of the implementation of aggregators and attacks. When the number of byzantine workers
is larger than 0, the attack API allows to return any custom gradient based on the gradient of non-byzantine workers.
In this framework, we first calculate the gradient of honest workers based on the current data sample, then we provide
these gradients to the byzantine work and retrieve the attack gradient, finally we pass all the honest and byzantine
gradients to the aggregator. The aggregator produces the final aggregated gradient which will be applied to the model.

### Attacker

All the attacks are implemented in [dbqf/attacks.py](https://github.com/Tabrizian/mix-tailor/blob/master/dbqf/attacks/attacks.py).
To implement your own custom attacks you only need to extend the `Attack` class and implement the `grad` function. The `grad` function
receives a list of honest gradients and has to return the attack gradient.

### Aggregator

The aggregator receives a pool of graidents where some of them come from honest workers and other ones come from byzantine workers.
The main job of the aggregator is to find out which aggregators are honest and which ones are not and produce the final gradient based on that.
The gradient produced by the aggregator will be applied to the model. All the aggregators are implemented in [dbqf/aggregators.py](https://github.com/Tabrizian/mix-tailor/blob/master/dbqf/aggregators/aggregators.py). In order to implemented your own aggregator, you need to 
extend the `Aggregator` class and implement the `agg_g` function. This function receives a list of gradients (some honest, some byzantine) and 
must return a tensor with the same size that contains the aggregated final gradient.


# Citation

```bibtex
@article{ramezani2022mixtailor,
  title={MixTailor: Mixed Gradient Aggregation for Robust Learning Against Tailored Attacks},
  author={Ramezani-Kebrya, Ali and Tabrizian, Iman and Faghri, Fartash and Popovski, Petar},
  journal={arXiv preprint arXiv:2207.07941},
  year={2022}
}
```
