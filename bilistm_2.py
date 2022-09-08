import glob
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.utils import pad_sequences, to_categorical
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense

files = glob.glob('./ner/*.tags')
data_pd = pd.concat([pd.read_csv(f, header=None, names=['text', 'label', 'pos']) for f in files], ignore_index=True)

print(data_pd.info())

# This class allows to vectorize a text corpus, by turning each text into either a sequence of integers (each integer
# being the index of a token in a dictionary) or into a vector where the coefficient for each token could be binary,
# based on word count, based on tf-idf...
text_tok = Tokenizer(filters='[\\]^\t\n', lower=False, split=' ', oov_token='<OOV>')
pos_tok = Tokenizer(filters='\t\n', lower=False, split=' ', oov_token='<OOV>')
ner_tok = Tokenizer(filters='\t\n', lower=False, split=' ', oov_token='<OOV>')

text_tok.fit_on_texts(data_pd['text'])
pos_tok.fit_on_texts(data_pd['pos'])
ner_tok.fit_on_texts(data_pd['label'])

ner_config = ner_tok.get_config()
text_config = text_tok.get_config()
print(ner_config)
# print(text_config)

# eval convert string to dictionary
text_vocab = eval(text_config['index_word'])
print("Unique words in vocab:", len(text_vocab))

ner_vocab = eval(ner_config['index_word'])
print("Unique NER tags in vocab:", len(ner_vocab))

# Transforms each text in texts to a sequence of integers.
x_tok = text_tok.texts_to_sequences(data_pd['text'])
y_tok = ner_tok.texts_to_sequences(data_pd['label'])

# O B-gpe O O O O O O O O O O O O O B-geo O B-tim O O O O O O O
# [2, 9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 4, 2, 2, 2, 2, 2, 2, 2]
print(data_pd['label'][0], y_tok[0])

# An Iraqi court has sentenced 11 men to death for the massive truck bombings in Baghdad last August that killed more than 100 people .
# [316, 89, 233, 13, 1112, 494, 240, 7, 248, 12, 2, 913, 1485, 528, 5, 146, 61, 570, 16, 38, 50, 55, 671, 39, 3]
print(data_pd['text'][0], x_tok[0])

max_len = 50
# padding, String, "pre" or "post" (optional, defaults to "pre"): pad either before or after each sequence.
x_pad = pad_sequences(x_tok, padding='post', maxlen=max_len)
y_pad = pad_sequences(y_tok, padding='post', maxlen=max_len)

print(x_pad.shape, y_pad.shape)

# Since there are multiple labels, each label token needs to be one-hot encoded like so:
num_classes = len(ner_vocab) + 1
Y = to_categorical(y_pad, num_classes)

# (62010, 50, 19)
print(Y.shape)

# B-gpe => 9 => [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
print('finally covert ', ner_vocab.get('9'), '=>', y_pad[0][1], '=>', Y[0][1])

vocab_size = len(text_vocab) + 1
embedding_dim = 64
rnn_units = 100
BATCH_SIZE = 90
num_classes = len(ner_vocab) + 1

dropout = 0.2


# None means it is a dynamic shape. It can take any value depending on the batch size you choose.
# num_units in TensorFlow is the number of hidden states, Positive integer, dimensionality of the output space.
# TimeDistributed , This wrapper allows to apply a layer to every temporal slice of an input.
# kernel_initializer Initializer for the kernel weights matrix, used for the linear transformation of the inputs
def build_model_bilstm(vocab_size, embedding_dim, rnn_units, batch_size, classes):
    return Sequential([
        Embedding(vocab_size, embedding_dim, mask_zero=True, batch_input_shape=[batch_size, None]),
        Bidirectional(LSTM(units=rnn_units, return_sequences=True, dropout=dropout, kernel_initializer=tf.keras.initializers.he_normal())),
        TimeDistributed(Dense(rnn_units, activation='relu')),
        Dense(classes, activation='softmax')
    ])


model = build_model_bilstm(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size=BATCH_SIZE, classes=num_classes)
print(model.summary())
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

X = x_pad

total_sentences = ner_config.get('document_count')
train_size = round(round(total_sentences / BATCH_SIZE) * 0.8) * BATCH_SIZE
test_size = round(round(total_sentences / BATCH_SIZE) * BATCH_SIZE)
print(train_size, test_size)

X_train = X[0: train_size]
Y_train = Y[0: train_size]

X_test = X[train_size: test_size]
Y_test = Y[train_size: test_size]

model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=15)

model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)

y_pred = model.predict(X_test, batch_size=BATCH_SIZE)

# convert prediction one-hot encoding back to number
y_pred = tf.argmax(y_pred, -1)
y_pnp = y_pred.numpy()

# convert ground true one-hot encode back to number
y_ground_true = tf.argmax(Y_test, -1)
y_ground_true_pnp = y_ground_true.numpy()

for i in range(10):
    x = 'sentence=> ' + text_tok.sequences_to_texts([X_test[i]])[0]
    ground_true = 'ground_true=> ' + ner_tok.sequences_to_texts([y_ground_true_pnp[i]])[0]
    prediction = 'prediction=> ' + ner_tok.sequences_to_texts([y_pnp[i]])[0]
    template = '|'.join(['{' + str(index) + ': <15}' for index, x in enumerate(x.split(' ')) if x != '<OOV>'])
    print(template.format(*x.split(' ')))
    print(template.format(*ground_true.split(' ')))
    print(template.format(*prediction.split(' ')))
    print('\n')
