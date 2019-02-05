## Decision Rules in Neural Networks

A semantic segmentation network with a softmax output layer can be seen as a statistical model that provides for each pixel of one image a probability distribution on pre-defined semantic class labels, given some weights and the input data. The predicted class in one pixel is then usually obtained by the maximum a-posteriori probability. In this way, the chance of an incorrect class estimation is minimized which is equivalent to the **Bayes** rule from decision theory. On the contrary, another mathematically natural approach is applying the **Maximum Likelihood** (ML) rule which maps features to the class with the largest conditional likelihood. The latter rule aims at finding the class for which given patterns are most typical (according to observed features in the training set), independent of the a-priori probability of the particular class. Consequently, more rare class objects can be detected by neural networks that might be biased due to training on unbalanced data.

## Result

In our experiments, we train one FRRN network (https://arxiv.org/pdf/1611.08323.pdf) from scratch using a proprietary and highly unbalanced dataset containing 20,000 annotated frames of video sequences recorded from street scenes. We adopt a localized method by computing the priors pixel-wise and compare the performance of applying the ML rule instead of the Bayes rule. The evaluation on our test set shows an increase of average recall with regard to instances of pedestrians and info signs by 25% and 23.4\%, respectively. In addition, we significantly reduce the non-detection rate for instances of the same classes by 61% and 38%. Link to the corresponding paper: https://arxiv.org/abs/1901.08394.

## This repository

... contains python scripts that produce segmentations with the Bayes & ML rule and the analysis tools in order to study the impact of the different decision rules. 

## Preparation

We suggest that the user places all the required input data in the folder called _"data/"_. It should contain:

- directory of training ground truth images
- directory of test ground truth images
- directory of test raw input images
- directory of frozen graph model (.pb)

In order to use own class labels modify _"labels.py"_. Set global variables by editing _"globals.py"_.

## Packages and their versions we used

- matplotlib==2.0.2
- numpy==1.13.3
- Pillow==4.3.0
- scikit-image==0.14.1
- simplejson==3.8.2
- sklearn==0.0
- tabulate==0.8.2
- tensorflow-gpu==1.9.0

See also _"requirements.txt"_.

## Run scripts

We used Python 3.4.6. Execute:

```sh
./run.sh
```

## Author
- Robin Chan, University of Wuppertal
