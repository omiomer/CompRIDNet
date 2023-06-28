# CompRIDNet
This is an unofficial implementation of a (compact) RidNet model and its comparison against UNet and DnCNN in image denoising

In this project, we experimented with state-of-the-art neural network architectures used in computer vision using an open-access image competition dataset where the aim is to deblur/denoise the data. We used RIDNet, a CNN architecture, as the core of our algorithm as it is the state-of-the-art algorithm for CNN models in image denoising (at the time of the implementation of this project) and we made it more compact using an autoencoder architecture. This allowed us to run the model with limited resources and minimal quality loss. We also test this model on the MNIST dataset to verify the integrity of the models in the preliminary tests and compare its performance with the original dataset. We compare our method with famous architectures to present our results.

The results could be replicated with an image library of choice.


![Architecture of the Main Model]([https://github.com/omiomer/CompRIDNet/blob/main/graphson3.pdf])
