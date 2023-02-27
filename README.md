# IIT-Gn_tasks
Q1. [5 Marks] Implement Inverse CDF sampling for the Fréchet distribution. With sufficient
number of samples, plot the kernel density estimation plot and show that you are able to
match the PDF. Also, reproduce the following figure and visualize it:

Q2. [5 Marks] Briefly explain and implement from scratch the following functions: i)
cross-entropy; ii) entropy; iii) mutual information; iv) conditional entropy; v) KL
divergence. Take appropriate example toy data/distributions and explain the insights
from calculating these quantities.

Q3.  [10 Marks] Download a week of PM2.5 dataset from the OpenAQ website for Delhi. Fill
in the missing data with appropriate methods. Use i) the sklearn Random Forest
regressor; ii) sklearn Linear regression and iii) Gaussian process regressor to interpolate
the PM2.5 values across the space. Use all the available features that you can get from
the OpenAQ website or elsewhere (e.g., meteorological variables). Compare the results
and prepare a table showing the metrics in K-Fold cross-validation setting.

Q4. [10 Marks] Prepare your own image dataset for binary classification with two classes:
Fox v/s Dog. Use an appropriate neural network library (JAX or any other library) and a
small CNN (not necessary to use big models like ResNet, or VGG if you do not have the
compute power). Use an appropriate active learning algorithm from
https://modal-python.readthedocs.io/en/latest/ library and show the active learning
iterations v/s test accuracy curve. Explain your insights.

Q5. within 200 words:
(https://arxiv.org/pdf/1906.08158.pdf)

Q6. [5 Marks] Consider a two-mode distribution with a Mixture of Gaussians (both modes
should have different heights and locations). Your task is to fit a single Normal
distribution to this distribution. Use KL-divergence to minimize the distance between
these two distributions. Create an animation that shows the iteration-wise progress as
you fit the normal distribution to the two-mode distribution. You have to show this for
Forward KL and Reverse KL divergence. Explain your insights about the nature of the fit
in each of these cases. You can use https:(//github.com/tensorflow/probability) JAX
substrate to do this task.

Q7. [15 Marks] You have a box of coins where any randomly chosen coin follows a
Kumaraswamy distribution with a=2, b=3. You pick a coin at random and toss it ten times
which yields = [H, T, H, H, T, H, H, H, H, H]. Given this data, what would be the
probability distribution of the coin that we picked up?
a. Try to find the answer analytically using the Bayes rule, and if the posterior is
available in the closed-form, visualize the resultant distribution. If the posterior
distribution is not available analytically, show the calculations up to the step
where it becomes intractable to solve further [5 Marks]
b. Use blackjax library with NUTS sampler to find the posterior distribution and plot
the posterior distribution [5 Marks].
c. Implement variational inference from scratch for this problem and plot the
posterior distribution. Visualize the fitting process v/s iterations. You are free to
assume the hyperparameters of variational inference but state them clearly [5
Marks].
State your insights from the comparison of fit from a., b., and c. Which one is
better in terms of fit? Which one is slower to compute?
