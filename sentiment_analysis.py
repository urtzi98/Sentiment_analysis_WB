import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D, Conv1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint

TAG_RE = re.compile(r'<[^>]+>')
DATA_PATH = "IMDB Dataset.csv"
MAX_LEN = 100
checkpoint_filepath = 'checkpoint/weights'

model_checkpoint_callback = ModelCheckpoint(
    filepath = checkpoint_filepath,
    save_weights_only = True,
    monitor = 'val_acc',
    mode = 'max',
    save_best_only = True
)

def read_data(path):
    # Kaggle dataset of 50K Moview Reviews
    movie_reviews = pd.read_csv(path)
    movie_reviews.isnull().values.any()
    
    return movie_reviews


def remove_tags(text):

    return TAG_RE.sub('', text)


def preprocess_text(review):
    review = remove_tags(review)
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = re.sub(r"\s+[a-zA-Z]\s+", ' ', review)
    review = re.sub(r'\s+', ' ', review)

    return review

def create_train_test(movie_reviews):
    X = []
    reviews = list(movie_reviews['review'])
    for review in reviews:
        X.append(preprocess_text(review))
    y = movie_reviews['sentiment']
    y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    return X_train, X_test, y_train, y_test

def prepare_embedding_layer(X_train, X_test):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    vocab_size = len(tokenizer.word_index) + 1 # unique words of the corpus
    X_train = pad_sequences(X_train, padding='post', maxlen=MAX_LEN)
    X_test = pad_sequences(X_test, padding='post', maxlen=MAX_LEN)

    embeddings_dictionary = dict()
    glove_file = open('glove.6B.100d.txt', encoding="utf8")

    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()

    embedding_matrix = np.zeros((vocab_size, 100))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    return X_train, X_test, embedding_matrix, vocab_size

def create_model(vocab_size, embedding_matrix):
    model = Sequential()
    embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=MAX_LEN , trainable=False)
    model.add(embedding_layer)

    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    # print(model.summary())

    return model


def train_cnn(X_train, y_train, X_test, y_test, model):
    # In each epoch the actual hyperparameters will be tested with the validation se
    out_values = model.fit(X_train, y_train, batch_size=128, epochs=6,
                           callbacks=[model_checkpoint_callback],
                           verbose=1,
                           validation_split=0.2)

    # score_train = model.evaluate(X_train, y_train, verbose=1)
    score_test = model.evaluate(X_test, y_test, verbose=1)
    # print("Train Score:", score_train[0])
    # print("Train Accuracy:", score_train[1])
    print("Test Score:", score_test[0])
    print("Test Accuracy:", score_test[1])
    # print_graphs(out_values)

    return model

def print_graphs(out_values):
    plt.plot(out_values.history['acc'])
    plt.plot(out_values.history['val_acc'])

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc = 'upper left')
    plt.show()

    plt.plot(out_values.history['loss'])
    plt.plot(out_values.history['val_loss'])

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc = 'upper left')
    plt.show()

def process_instance(instance, X_train):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    instance = tokenizer.texts_to_sequences(instance)

    # flat_list = []
    # for sublist in instance:
    #     for item in sublist:
    #         flat_list.append(item)

    # flat_list = [flat_list]

    instance = pad_sequences(instance, padding='post', maxlen=MAX_LEN)

    return instance

def main():
    movie_reviews = read_data(DATA_PATH)
    X_train, X_test, y_train, y_test = create_train_test(movie_reviews)
    X_train, X_test, embedding_matrix, vocab_size = prepare_embedding_layer(X_train, X_test)
    # embedding matrix is the initial weights configuration
    model = create_model(vocab_size, embedding_matrix)
    # model = train_cnn(X_train, y_train, X_test, y_test, model)
    #model.load_weights(checkpoint_filepath)
    # prediction on single reviews using the best weights
    score_test = model.evaluate(X_test, y_test, verbose=1)
    print("Test Score:", score_test[0])
    print("Test Accuracy:", score_test[1])
    print(np.count_nonzero(y_test==1))
    print(np.count_nonzero(y_test==0))
    print(y_test.shape)

if __name__ == '__main__':
    main()

# https://stackabuse.com/python-for-nlp-movie-sentiment-analysis-using-deep-learning-in-keras/
# https://medium.com/datadriveninvestor/deep-learning-lstm-for-sentiment-analysis-in-tensorflow-with-keras-api-92e62cde7626 
# https://www.kaggle.com/ngyptr/lstm-sentiment-analysis-keras 
# https://realpython.com/python-keras-text-classification/ 
# https://github.com/PacktPublishing/Sentiment-Analysis-through-Deep-Learning-with-Keras-and-Python 
