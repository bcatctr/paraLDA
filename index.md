## Summary
We are going to implement a distributed, parallel LDA (latent dirichlet allocation) algorithm for building topic model on large corpus using the OpenMPI library and CUDA parallel programming model. We will parallelize the original sequential algorithm over a cluster of GPUs.
## Background
In natural language processing, latent Dirichlet allocation (LDA) is a generative statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar. For example, if observations are words collected into documents, it posits that each document is a mixture of a small number of topics and that each word's creation is attributable to one of the document's topics. 
	
### Generative process
At the beginning, we will initialize the XXX distributions with uniform distribution.

For each iteration, we need to ...
TODO

## Challenge
The original algorithm is totally sequential. There are mainly three parts take up the most computation time so they are what we need to parallelize.

* Gibbs sampling is by definition sequential. Every choice of z depends on all the other z's (TODO: what is z here?). We need to modify the algorithm and try approximate distributed LDA.
* Update the distribution (please give it a fancy name) involves many independent computation steps and finally a normalization over all of them. We need to fully parallelize this part.
* Faster sampling is the operation of assigning each word a topic. This operation will be called in a very high frequency. Though this operation will not be parallelized, we still need to find faster implementation.

The basic idea is to split the gibbs sampling into different machines and each machine perform parallelized Update the distribution (please give it a fancy name) locally. Each machine will treat their local distribution as global distribution, and communicate to exchange their updates for every one or several iterations. This method has these challenges:

* How to split the gibbs sampling.
* Choose which synchronization model.
* How to parallelize Update the distribution (please give it a fancy name) over GPUs.
* How to minimize the communication overheads between
	- Different machines
	- CPU and GPU


## Resources
We plan to use latedays cluster for testing our system. The latedays cluster supports OpenMPI and CUDA which meets our requirements. The possible problem is that there maybe many other users using this cluster and have influence on our performance.
## Goals and Delieverables
Plan to achieve:

* An OpenMPI and CUDA based, distributed and parallel LDA implementation.
* Implement banchmark based on Standford NLP toolkits.
* Conduct experiments on NYTimes dataset.
* Achieve high performance (Can we have a expect value here?) over banchmark results.

Hope to achieve:

* Compare our system with state-of-art parallel LDA systems like Yahoo LDA, lightlda, etc. and achieve similar results.

## Platform choice
As mentioned before, we plan to use latedays clusters.
## Schedule
|Dates    |Planned Goals                                                                      |
|:--------|:----------------------------------------------------------------------------------|
|4/10-4/16|Prepare dataset; Build banchmark system; Read related papers and make system design|
|4/17-4/23|Implement distrubted version; Prepare for checkpoint report                        |
|4/24-4/30|Implement parallelism on each machine; Implement faster sampling                   |
|5/1-5/8  |Optimize the results and conduct experiments                                       |
|5/9-5/12 |Prepare for final report                                                           |
## References
[1] Blei, David M., Andrew Y. Ng, and Michael I. Jordan. "Latent dirichlet allocation." Journal of machine Learning research 3.Jan (2003): 993-1022.
[2] Newman, David, et al. "Distributed algorithms for topic models." Journal of Machine Learning Research 10.Aug (2009): 1801-1828.
[3] Yao, Limin, David Mimno, and Andrew McCallum. "Efficient methods for topic model inference on streaming document collections." Proceedings of the 15th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2009.
[4] Li, Aaron Q., et al. "Reducing the sampling complexity of topic models." Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2014.
[5] Yu, Hsiang-Fu, et al. "A scalable asynchronous distributed algorithm for topic modeling." Proceedings of the 24th International Conference on World Wide Web. ACM, 2015.
[6] Yuan, Jinhui, et al. "Lightlda: Big topic models on modest computer clusters." Proceedings of the 24th International Conference on World Wide Web. ACM, 2015.