import glob
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical
from keras import Sequential
import pandas as pd
from keras.layers import Embedding, Bidirectional, LSTM, Dense
import re

files = glob.glob('./nlu_data/SMSSpamCollection.csv')
data_pd = pd.concat([pd.read_csv(f, header=None, names=['label', 'text'], sep='\t') for f in files], ignore_index=True)

print(data_pd.info())

text_tok = Tokenizer(lower=False, split=' ', oov_token='<OOV>')
label_tok = Tokenizer(lower=False, split=' ', oov_token='<OOV>')

text_tok.fit_on_texts(data_pd['text'])
label_tok.fit_on_texts(data_pd['label'])

text_config = text_tok.get_config()
label_config = label_tok.get_config()

print(text_config.get('document_count'))
print(label_config)

text_vocab = eval(text_config['index_word'])
label_vocab = eval(label_config['index_word'])

x_tok = text_tok.texts_to_sequences(data_pd['text'])
y_tok = label_tok.texts_to_sequences(data_pd['label'])

print('text', data_pd['text'][0], x_tok[0])
print('label', data_pd['label'][0], y_tok[0])

max_len = 172

x_pad = pad_sequences(x_tok, padding='post', maxlen=max_len)
y_pad = y_tok

num_classes = len(label_vocab) + 1
Y = to_categorical(y_pad, num_classes)

vocab_size = len(text_vocab) + 1
embedding_dim = 64
rnn_units = 100
BATCH_SIZE = 90
dropout = 0.2

model = Sequential([
    Embedding(vocab_size, embedding_dim, mask_zero=True, batch_input_shape=[BATCH_SIZE, None]),
    Bidirectional(LSTM(units=rnn_units, return_sequences=True, dropout=dropout, kernel_initializer=tf.keras.initializers.he_normal())),
    Bidirectional(LSTM(round(num_classes / 2))),
    Dense(num_classes, activation='softmax')
])

print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

X = x_pad
# 5572  61 =  49 + 12
X_train = X[0: 4410]
Y_train = Y[0: 4410]

print(Y_train.shape)

X_test = X[4410: 5490]
Y_test = Y[4410: 5490]

model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=15)

model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)

y_pred = model.predict(X_test, batch_size=BATCH_SIZE)

# 3s 43ms/step - loss: 0.2169 - accuracy: 0.9333

# convert prediction one-hot encoding back to number
y_pred = tf.argmax(y_pred, -1)
y_pnp = y_pred.numpy()

# convert ground true one-hot encode back to number
y_ground_true = tf.argmax(Y_test, -1)
y_ground_true_pnp = y_ground_true.numpy()


for i in range(20):
    x = 'sentence=> ' + text_tok.sequences_to_texts([X_test[i]])[0]
    x = re.sub(r'<OOV>*.', '', x)
    ground_true = 'ground_true=> ' + label_tok.sequences_to_texts([[y_ground_true_pnp[i]]])[0]
    prediction = 'prediction=> ' + label_tok.sequences_to_texts([[y_pnp[i]]])[0]
    print(x)
    print(ground_true)
    print(prediction)
    print('\n')
