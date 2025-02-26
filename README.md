# Classification Tutorial
Image classification is the computer vision task of mapping an image to a class label.

## Tutorial - Training and running a neural network
### Learning Objectives
- Work with git on Google Colab
- Write a NeuralNetwork class
- Implementing training loop
- Implementing evaluation (run/inference) loop
- Identify and import loss function
- Identify and import evaluation metrics
- Coding practices, documentation

Files involved:
```
src/classification_nn.py
src/networks.py
src/run_classification_nn.py
src/train_classification_nn.py
src/log_utils.py
```

### Tasks
- Set up git on Google Colab
- Clone repository on your local machine and Google Colab
- Create a branch from master using your-name
- Develop on your local machine and push to your branch in remote
- Implement all TODOs in the files specified above
- Commits should be pushed to your remote branch on github
- Checkout your branch on Google Colab, fetch and rebase, and test your code using run commands (see `bash` folder)
- Report your scores below (you should get a number higher than 90% on MNIST and 51% on CIFAR-10) i.e.
```
Mean accuracy over 10000 images: 91.370% (MNIST)
Mean accuracy over 10000 images: 52.840% (CIFAR-10)
```

```
My accuracy scores:
Mean accuracy over 10000 images: 96.630% (MNIST)
Mean accuracy over 10000 images: 51.850% (CIFAR-10)
```

## Tutorial - Training and running a convolutional neural network (CNN)
### Learning Objectives
- Work with git on Google Colab
- Write a ResNet18 encoder network
- Write a VGGNet11 encoder network
- Write a ClassificationModel class that can initatiate different types of CNN
- Implement forward, compute_loss, saving, restoring, etc. functions for ClassificationModel
- Implement Tensorboard logging
- Implementing training loop
- Implementing evaluation (run/inference) loop
- Identify and import loss function
- Identify and import evaluation metrics
- Coding practices, documentation

Files involved:
```
src/classification_cnn.py
src/classification_model.py
src/net_utils.py
src/networks.py
src/run_classification_cnn.py
src/train_classification_cnn.py
```

### Tasks
- Develop on your local machine and push to your branch in remote
- Implement all TODOs in the files specified above
- Commits should be pushed to your remote branch on github
- Checkout your branch on Google Colab, fetch and rebase, and test your code using run commands (see `bash` folder)
- Report your scores below (you should get a number higher than 98% on MNIST and 80% on CIFAR-10) i.e.
```
Mean accuracy over 10000 images: 98.240% (VGG-11 on MNIST)
Mean accuracy over 10000 images: 80.210% (VGG-11 on CIFAR-10)

Mean accuracy over 10000 images: 99.270% (ResNet-18 on MNIST)
Mean accuracy over 10000 images: 81.340% (ResNet-18 on CIFAR-10)
```

```
my scores
Mean accuracy over 10000 images: 98.950% (ResNet-18 on MNIST)
Mean accuracy over 10000 images: 81.200% (ResNet-18 on CIFAR-10)

Mean accuracy over 10000 images: 99.020% (VGG-11 on MNIST)
Mean accuracy over 10000 images: 85.630% (VGG-11 on CIFAR-10)
```