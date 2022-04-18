# Geo-DefakeHop: High-Performance Geographic Fake Image Detection

Still Constructing...

![Framework](img/framework.png)

A robust fake satellite image detection method, called ["Geo-DefakeHop"](https://arxiv.org/abs/2110.09795), is proposed in this work. Geo-DefakeHop is developed based on the parallel subspace learning (PSL) methodology. PSL maps the input image space into several feature subspaces using multiple filter banks. By exploring response differences of different channels between real and fake images for filter banks, Geo-DefakeHop learns the most discriminant channels based on the validation dataset, uses their soft decision scores as features, and ensemble them to get the final binary decision. Geo-DefakeHop offers a light-weight high-performance solution to fake satellite images detection. The model size of Geo-DefakeHop ranges from 0.8K to 62K parameters depending on different hyper-parameter setting. Experimental results show that Geo-DefakeHop achieves F1-scores higher than 95% under various common image manipulations such as resizing, compression and noise corruption.


## Cite us
If you use this repository, please consider to cite.
```
@article{chen2021geo,
  title={Geo-DefakeHop: High-Performance Geographic Fake Image Detection},
  author={Chen, Hong-Shuo and Zhang, Kaitai and Hu, Shuowen and You, Suya and Kuo, C-C Jay},
  journal={arXiv preprint arXiv:2110.09795},
  year={2021}
}
```
## Acknowledgment
This work was supported by the Army Research Labora- tory (ARL) under agreement W911NF2020157. Computation for the work was supported by the University of Southern California’s Center for High Performance Computing (hpc.usc.edu).
