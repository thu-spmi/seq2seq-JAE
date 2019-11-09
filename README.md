

Source code for the paper:

**Yunfu Song, Zhijian Ou. "Semi-supervised Seq2seq Joint-stochastic-approximation Autoencoders with Applications to Semantic Parsing"** from [SPMI Lab](http://oa.ee.tsinghua.edu.cn/~ouzhijian/software.htm), Tsinghua University, Beijing, China.

### Dependencies

1. python 2.7
2. pytorch 0.3.1

### Training

1. Prepare datasets

~~~
sh scripts/prepare_dataset.sh
~~~

This script will process datasets of ATIS and DJANGO, and split different parallel data size.

2. Semi-supervised training with JAE and VAE

~~~
sh scripts/atis/experiment.sh 1000 1
~~~

This script produces results on ATIS dataset, containing supervised training of encoder and decoder, LM training of p(z) and p(x), semi-supervised training of VAE, JAE and bi-JAE, 1000 the parallel data size, 1 the random seed.

Similarly, on  DJANGO dataset, run:
~~~
sh scripts/django/experiment.sh 1000 1
~~~
Note that when experimenting on DJANGO dataset, source code of Django (<https://github.com/django/django>) should be placed at data/django_code to train p(z).

Change parallel data size and random seed to get results of Table 2 in the paper.

### Acknowledgments

The code is modified on <https://github.com/pcyin/structVAE>.



