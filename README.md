# SpeechToText

**Purpose:** To convert a set of spoken English speech commands into text using a convolutional neural network model to predict the outcome given an audio sample input.

**Pre-reqs:** Make sure to download the dataset of training audio samples from kaggle (linked in sources below). Also pip install all the required libraries.

### Usage: 
"python train_cnn.py"
"python test_cnn.py"

### Preprocessing:
In this program we analyze a small dataset of audio samples from kaggle that are short recordings of a small set of words. We sample these recordings using librosa at a rate of 8000, following the Nyquist-Shannon sampling theorem (2x normal human speech frequency of ~4000Hz). In analytics.ipynb, we discover that not all the words have as many samples as each other, and that some of the audio files are under one second, while most are exactly 1s. To ensure clean and consistent data, we only consider those samples which are exactly 1s (vector size of 8000) long, and only the words which have enough training data. Once we have all our relevant audio files processed into vectors holding the values of the resulting digital signal following the librosa sampling, we can proceed to apply deep learning techniques to make predictions. 

### Model design:
The model we train to make predictions is a convolutional neural network, commonly used to find structures and patterns in input data to classify it with a label. Since audio data is quite complex and involves a lot of nuances, we design our model with a corresponding level of complexity, slowly cutting our data down until we can make a prediction of (in this case) 10 different label outputs. The model consists of four one-dimensional convolutional layers, and three fully connected layers, using the reLU activation function and a dropout layer (to reduce overfitting) in between each of the convolutional and dense layers.

### Training:
To train the model, we load our dataset and use pytorch's built in functions to perform a forward pass, and backpropogation to compute the updated weights and biases for each epoch of training. After each train step, we evaluate the performance of the model based on a few metrics (accuracy and AUC/ROC score) and log them. We also implement a stopping method to stop the training once we have reached a certain number (patience) of passed epochs where there has been no improvement to the global loss of the model's steps. Once the model is finished training we can select the model from the epoch with the best validation accuracy to test with.

### Testing:
To test the model, we load our dataset again and select the epoch from which to load the model. Then we can evaluate the metrics on the output of the model against the test data as we did for training and validation data during to measure the success of the model. There is also an optional audio mode that lets you record a voice command and uses the model to convert it to text.
Set of voice commands: "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"

### Sources:
Download dataset used: https://www.kaggle.com/competitions/tensorflow-speech-recognition-challenge/data
Analytics code from: https://www.analyticsvidhya.com/blog/2019/07/learn-build-first-speech-to-text-model-python/
