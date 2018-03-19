# Use LSTMs to perfom sentiment analysis

Here, Keras, word embeddings and LSTMs are used to detect sentiment within a text document. This was one of the assignments of a course I did on Deep Learning and Sequence Models. 

In the onventional methods of performing sentiment analysis, one needs to explicitly tell the model to focus on negative words such as 'not' (eg: 'not good' - a negative sentiment). The advantage of using LSTMs to perform the same task is that LSTMs by definition encode the information from the preceding words, and upon training for a reasonable number of epochs, can automatically "learn" the info from pairings that are labelled positive or negative.

The model architecture is as follows:

  1) Read in the input data, 'X' containing the different text documents, and Y containing the corresponding sentiment label, denoted by        an integer.
  2) Create a dictionary that maps all the unique words in the text corpora to a corresponding number, and replace the words in each            sentence with its number.
  3) Make the integers representing the sentiment into a one-hot encoded array of size equalling the number of classes (sentiments).
  4) Pass the sequences through an embedding layer that uses pre trained GloVe vectors. The resulting embedding matrix is passed on to an      LSTM layer that uses 300 hidden state units.
  5) Add a dropout layer (here with droppout of 0.5).
  6) Further pass the output to another LSTM and Dropout layer. and then pass it into a Dense layer with 'num_classes' units and softmax        activation.
  
  Fit the model with the given data and train it for sufficient number of epochs.
  
  The code is heavily documented, and will hopefully clear any lingering doubts.
  
