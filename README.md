# Description

Implementation of the [paper](https://arxiv.org/pdf/1612.01474v1.pdf) **Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles**

![Sinusoidal gaussian regressor](sinusoidal.png)
Predictive uncertainty estimates obtained by using the ensemble approach proposed in the paper. The value of the sinusoidal function for values between -4 and 4 was used in training and at test time, the trained model is used to predict function values between -8 and 8. The blue line represents the true function, red represents the mean and the other two curves represent (mean + 3*std)

The implementation is partly inspired from [this](https://github.com/muupan/deep-ensemble-uncertainty) repository

**Author** : Anirudh Vemula
