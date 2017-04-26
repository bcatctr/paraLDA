## Process Review
We have implemented a sequential CPU version for LDA so far. In the first week from April 10 to April 16, we prepared for testing datasets and preprocessed the datasets to be clean and consistent. We decided to use [**UCI Bag of Words Data Set**](https://archive.ics.uci.edu/ml/datasets/Bag+of+Words), [**UCI NIPS Conference Papers 1987-2015 Data Set**](https://archive.ics.uci.edu/ml/datasets/NIPS+Conference+Papers+1987-2015) and [**The 20 Newsgroups Data Set**](http://qwone.com/~jason/20Newsgroups/). Besides, we designed the framework for general usage of LDA.

In the second week from April 17 to April 23, we implemented the LDA framework and tested on three data sets abovementioned. We built the evaluation metrics in terms of iterations and time on Log Likelihood. Besides, we visualized the Topic Extraction process by text. Currently, we are working on implementing the distributed version for LDA.


## Goals and Delieverables
We have intended to build benchmark based on Stanford NLP toolkits. However, Stanford NLP is implemented in JAVA, which make the comparison with our C++ version of LDA unreliable. Thus, we decided to build our own sequential version benchmark. The NYTimes dataset is not suitable for evaluation for its raw texts and vocabulary. We use three data sets which have been syntactically preprocessed and have different magnitude of size, which can help in both horizontal and vertical comparison. Thus, out goals can be revised as:

* An OpenMPI and CUDA based, distributed and parallel LDA implementation.
* Explore the possibility of speedup in the aspect of algorithm approximation.
* Implement a sequential LDA framework as banchmark. Achieve high performance over banchmark results.
* Conduct experiments on **UCI Bag of Words Data Set**, **UCI NIPS Conference Papers 1987-2015 Data Set**, **The 20 Newsgroups Data Set**. Compare the performances both vertically and horizontally.

We still Hope to achieve:

* Achieve similar or better performance compareed with state-of-art parallel LDA systems like Yahoo LDA, lightlda, etc.


## Revised Schedule

|Dates        |Planned Goals                                                                      | Status |
|:---------   |:----------------------------------------------------------------------------------|:--------|
|4/10-4/16    |Prepare dataset; Preprocess datasets; Read related papers and make system design   |  Done  |
|4/17-4/23    |Implement sequential; Build Evaluation metrics; Prepare for checkpoint report      |  Done  |
|4/24-4/26    |Test the sequential version implementaion                                          |  Need to compare with <br> peer sequential version  |
|4/27-4/30    |Implement distributed version among nodes in latedays machine                      |    |
|5/1-5/4      |Implement thread parallelism on each worker; Sparse LDA Speedup                    |    |
|5/5-5/8      |Implement faster sampling; Optimize LDA initialization                             |    |
|5/9-5/12     |Prepare for final report          



## Concerned Issues
The faster sampling might need some advanced data structures and it will also sacrifice convergence speed, the tradeoff between convergence and efficiency needs to be further discussed. The GPU based LDA might incur extra cost in calling kernel functions and copying data between GPU and CPU, it might only help when the number of topic is large.