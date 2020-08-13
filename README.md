# UnsupervisedNN
This repository proposes the unsupervised training of
a feedforward neural network to solve parametric optimization
problems involving large numbers of parameters. Such unsupervised training, which consists in repeatedly sampling parameter
values and performing stochastic gradient descent, foregoes the
taxing precomputation of labeled training data that supervised
learning necessitates. As an example of application, we put this
technique to use on a rather general constrained quadratic program. Follow-up letters subsequently apply it to more specialized
wireless communication problems, some of them nonconvex in
nature. In all cases, the performance of the proposed procedure
is very satisfactory and, in terms of computational cost, its
scalability with the problem dimensionality is superior to that
of convex solvers.
