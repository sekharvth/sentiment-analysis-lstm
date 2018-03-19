
# import necessary packages
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM
from keras.layers.embeddings import Embedding
np.random.seed(1)

X_train, Y_train = # read data
# X_train is a Series or list containing the texts, and Y_train contains the index (an integer) of the sentiment of the corresponding text

# find out the maximum length of sentences in the text corpora
maxlen = len(max(X_train, key = len).split())

num_classes = len(Y_train.unique())

# make a dictionary of words and their corresponding indexes
word_to_index = {}
for pos, i in enumerate(X_train):
  word_to_index[i] = pos
 
# read the GloVe vectors file
def read_glove_vecs(file):
    with open(file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        
        for line in f:
            line = line.strip().split()
            word = line[0]
            words.add(word)
            word_to_vec_map[word] = np.array(line[1:], dtype=np.float64)
            
    return words, word_to_vec_map

words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

def sentences_to_indices(X, word_to_index, max_len):
    
    # number of training examples
    m = X.shape[0]                                   
    
    # initialise X_indices as zero matrix of shape (num_examples, max_len)
    X_indices = np.zeros((m, max_len))
    
    # loop over training examples
    for i in range(m):                               
        
        # Convert the 'i'th training sentence to lower case, and split into words
        sentence_words = X[i].lower().split()
        
        # set a counter variable for obtaining the next word in the sentence
        j = 0
        
        # loop over the words in sentence
        for w in sentence_words:
        
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            X_indices[i, j] = word_to_index[w]
            
            # prepare counter for next word in sequence
            j = j+1
    
    return X_indices
    
# make the input sentences into array sequences of their indices, and make the target indices(sentiment indices) into one-hot representations
X_train_indices = sentences_to_indices(X_train, word_to_index, maxlen)
Y_train_indices = np.eye(num_classes)[Y_train.reshape(-1)]

vocab_len = len(word_to_index)                 
emb_dim = 50     

# make empty embedding matrix to store 50-d word embeddings of the sentences
embedding_matrix = np.zeros((vocab_len, emb_dim))

# corresponding to the index of each row of embedding matrix, fill in the values of 50 dimensional word embedddings
for word,index in word_to_index.items():
    try:
        embedding_matrix[index,:] = word_to_vec_map[word]
    # if word is not present in GloVe vectors, that index position is already filled with zeros, as we had initialized
    # all rows to zero in the first place
    except:
        continue 
                 
# make a Keras embedding layer of shape (vocab_size, emb_dim) and set 'trainable' argument to 'True' if you want to train your
# own word embeddings on top of pre trained GloVe vectors(can improve performance, as we'll get embeddings more suited to current text corpus)
embed_layer = Embedding(input_dim = vocab_len, output_dim = emb_dim, trainable = False)

# build the embedding layer
embed_layer.build((None,))

# Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
embed_layer.set_weights([embedding_matrix])

# Start defining the Keras model
# Define sentence_indices as the input to the model, of shape (maxlen,) and dtype 'int32'
sentence_indices = Input(shape = (maxlen,) dtype = 'int32')

# Propagate sentence_indices through the embedding layer, to get the embeddings
embeddings = embed_layer(sentence_indices)  

# Propagate the embeddings through an LSTM layer with 300-dimensional hidden state
X = LSTM(300, return_sequences = True)(embeddings)

# Add dropout with a probability of 0.5
X = Dropout(0.5)(X)

# Propagate X trough another LSTM layer with 300-dimensional hidden state
X = LSTM(300)(X)

# Add dropout with a probability of 0.5
X = Dropout(0.5)(X)

# Propagate X through a Dense layer with softmax activation to get back a batch of 'num_classes' number of values
X = Dense(num_classes, activation = 'softmax')(X)

# create model instance 
model = Model(inputs = sentence_indices, outputs = X)

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the training data on the model, while shuffling the order of the training set
model.fit(X_train_indices, Y_train_indices, epochs = 50, batch_size = 32, shuffle=True)


