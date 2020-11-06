---
date: 2020-08-07
---

# neural networks

## Links
[]()
[A Gentle Introduction to the Rectified Linear Unit (ReLU)](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)

[How to Fix the Vanishing Gradients Problem Using the ReLU](https://machinelearningmastery.com/how-to-fix-vanishing-gradients-using-the-rectified-linear-activation-function/)

[How Do Convolutional Layers Work in Deep Learning Neural Networks?](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/)

[A Gentle Introduction to Pooling Layers for Convolutional Neural Networks](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/)

[Number of Parameters and Tensor Sizes in a Convolutional Neural Network (CNN)](https://www.learnopencv.com/number-of-parameters-and-tensor-sizes-in-convolutional-neural-network/)

[NLP course to look at](https://lena-voita.github.io/nlp_course.html#whats_inside_lectures)

[machine engineerin book(costs to much)](http://www.mlebook.com/wiki/doku.php)

[exploding gradients and clipping](https://machinelearningmastery.com/how-to-avoid-exploding-gradients-in-neural-networks-with-gradient-clipping/)



## Activation functions 

### Linear
No backpropagation(gradient descent)

### NonLinear
bold = most common
- **relu**
- **tanh**
- **sigmoid**
- softmax
- swish
- leaky relu

sigmoid and tanh saturate(values near -1 or 1 snap to -1 or 1 respectively)
relu -> g(z) = max{0, z}

## RNN

### parameters

- input size
- hidden size
- num layers
- nonlinearity
- bias
- batch fisrt
- dropout
- bidirectional

### inputs

- input of shape (seq len, batch, input size
- h 0 (num layers, num directions, batch, hidden size)

### outputs

- output of shape (seq len, batch, input size
- h n (num layers, num directions, batch, hidden size)


input1 (L, N, Hin)

input2 (S, N, Hout)

output1 (L, N, Hall) Hall=num_directions\*hidden_size

output2 (S, N, Hout)

# cnn

- **CNNs have a habit of overfitting**, even with pooling layers. **Dropout should be used** such as between fully connected layers and perhaps after pooling layers
- Pooling layer is responsible for passing on the values to the next and previous layers during forward and backward propagation respectively.
- 2d -> data (w, h)
- color image -> 3 channels red green and blue -> (w, h, c)


# Hyperparameters optimization

> improve the model

(ordered by usefulness)

- Bayesian Optimization
- Random search 
- Grid search


# Backpropagation

> In fitting a neural network, backpropagation computes the gradient of the loss function with respect to the weights of the network

[cross entropy loss function](https://en.wikipedia.org/wiki/Cross_entropy) (for classification)

[squared error loss(SEL) loss function](https://en.wikipedia.org/wiki/Mean_squared_error) (for regression)

# RNN

[rnn.md](rnn.md)
