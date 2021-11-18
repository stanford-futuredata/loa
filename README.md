# Finding Label and Model Errors in Perception Data With Learned Observation Assertions

This is the project page for [Finding Label and Model Errors in Perception Data With Learned Observation Assertions](https://ddkang.github.io/papers/2021/loa-vldb-workshop.pdf).

Please read the [paper](https://ddkang.github.io/papers/2021/loa-vldb-workshop.pdf) for full technical details.


# Installation

In the root directory, run
```
pip install -e .
```


# Examples

We provide an example of the Lyft Level 5 percetion dataset.
We have provided model predictions for convenience, but you will need to download the dataset [here](https://level-5.global/data/perception/).

All of the scripts are available in `examples/lyft_level5`. In order to run the scripts, do the following:
1. Set the data directories in `constants.py`.
2. Learn the priors with `learn_priors.py`.
3. Run LOA with `prior_lyft.py`.

You can visualize the results with `viz_track.py`.



# Citation 

If you find this project useful, please cite us at
```
@article{kang2021finding,
  title={Finding Label and Model Errors in Perception Data With Learned Observation Assertions},
  author={Kang, Daniel and Arechiga, Nikos and Pillai, Sudeep and Bailis, Peter and Zaharia, Matei},
}
```
and contact us if you deploy LOA!
