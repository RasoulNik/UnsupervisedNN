Welcome to the UnsupervisedNN wiki!
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

## Code structure
The codes related to the cellular and cell-free applications are located in respective folders. Also, inside each folder, the uplink and downlink scenarios are separated. For example, the uplink of the cell-free is located in the file ../Cell-free/uplink.py.

Inside each script, you can specify train and load mode. If you set load = 1, it will load from the pre-trained model, while, load = 0 will train a new model. Trained models are going to be saved in the models folder with time tags.

### GPU support
if you have GPU  and the proper version of the of TensorFlow  (for example if you are using a TF container with the GPU tag) in your system, you can set GPU_mode = 1 and the code will run much faster.
### How can I use this library for my own application?
The code is modular. You need to define three classes.
1. Loss class: It specifies how good is the output of the NN for the given input.
2. NN class: The design of this class is very straight forward. Based on the dimension of the input and output you design a fully connect NN. The dimension of the input layer should be bigger than the dimension of the input dimension and the output layer dimension is determined by the output size. The size of the hidden layer should be between these two limits.
3. UNN class: This class puts the previous classes together and outputs the cost or loss for the given parameters (input).

## Citing
To site this repository please using the following papers:

R. Nikbakht, A. Jonsson and A. Lozano, "Unsupervised Learning for Parametric Optimization," in IEEE Communications Letters, doi: 10.1109/LCOMM.2020.3027981.
URL: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9210010&isnumber=5534602

R. Nikbakht, A. Jonsson and A. Lozano, "Unsupervised Learning for Cellular Power Control," in IEEE Communications Letters, doi: 10.1109/LCOMM.2020.3027994. systems}.
URL: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9210079&isnumber=5534602

R. Nikbakht, A. Jonsson and A. Lozano, "Unsupervised Learning for C-RAN Power Control and Power Allocation," in IEEE Communications Letters, doi: 10.1109/LCOMM.2020.3027991.
URL: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9210049&isnumber=5534602
