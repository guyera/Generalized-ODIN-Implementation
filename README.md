# Generalized-ODIN-Implementation

This is a (reproduced) implementation of the method described in [this paper](https://arxiv.org/pdf/2002.11297.pdf). Note that this implementation is not currently capable of perfectly reproducing the results published in the paper, though the results are close; the reason for this is unknown. If you would like to contribute, please see the contribution section below.

## Running the demo

Defaults: cosine h(x), denseNet, cifar10, imagenet crop, 300 epochs, cross-entropy loss

Main two files are cal.py and deconfnet.py

To train, simply call
```python cal.py ```

There are many arguments which may be supplied when running the program. However, this project is still very much a work in progress; feel free to experiment with them at your discretion.

## Contributions
If you would like to contribute to the project, feel free to open issues and make pull requests. While the maintainers of this repository are often busy with other research, and while updates may be infrequent, this repository is still actively maintained; your issues and pull requests will not go unnoticed.
