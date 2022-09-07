>创建一个简单的模型理解句子某些词的语义（NER）

### 加载一些包
```python
import glob
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.utils import pad_sequences, to_categorical
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense
````
### 加载标签和语句
在`ner`文件夹里面有一堆原始数据，每句话后面是每个词的标签。
```
example :
An Iraqi court has sentenced 11 men to death for the massive truck bombings in Baghdad last August that killed more than 100 people .,O B-gpe O O O O O O O O O O O O O B-geo O B-tim O O O O O O O,DT JJ NN VBZ VBN CD NNS TO NN IN DT JJ NN NNS IN NNP JJ NNP WDT VBD JJR IN CD NNS .
```
```python
files = glob.glob('./ner/*.tags')
data_pd = pd.concat([pd.read_csv(f, header=None, names=['text', 'label', 'pos']) for f in files], ignore_index=True)

print(data_pd.info())
```

### 序列化文本

首先把文本和标签`Token`化
```
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
```
这里打印了标签的Token信息，出现率越高的词索引会比较小。
```bash
{'num_words': None, 'filters': '\t\n', 'lower': False, 'split': ' ', 'char_level': False, 'oov_token': '<OOV>', 'document_count': 62010, 'word_counts': '{"O": 1146068, "B-gpe": 20436, "B-geo": 48876, "B-tim": 26296, "I-tim": 8493, "B-org": 26195, "I-org": 21899, "B-per": 21984, "I-per": 22270, "I-geo": 9512, "B-art": 503, "B-nat": 238, "B-eve": 391, "I-eve": 318, "I-art": 364, "I-gpe": 244, "I-nat": 62}', 'word_docs': '{"B-geo": 31660, "B-gpe": 16565, "B-tim": 22345, "O": 61999, "B-org": 20478, "I-org": 11011, "I-tim": 5526, "B-per": 17499, "I-per": 13805, "I-geo": 7738, "B-art": 425, "B-nat": 211, "B-eve": 361, "I-eve": 201, "I-art": 207, "I-gpe": 224, "I-nat": 50}', 'index_docs': '{"3": 31660, "9": 16565, "4": 22345, "2": 61999, "5": 20478, "8": 11011, "11": 5526, "7": 17499, "6": 13805, "10": 7738, "12": 425, "17": 211, "13": 361, "15": 201, "14": 207, "16": 224, "18": 50}', 'index_word': '{"1": "<OOV>", "2": "O", "3": "B-geo", "4": "B-tim", "5": "B-org", "6": "I-per", "7": "B-per", "8": "I-org", "9": "B-gpe", "10": "I-geo", "11": "I-tim", "12": "B-art", "13": "B-eve", "14": "I-art", "15": "I-eve", "16": "I-gpe", "17": "B-nat", "18": "I-nat"}', 'word_index': '{"<OOV>": 1, "O": 2, "B-geo": 3, "B-tim": 4, "B-org": 5, "I-per": 6, "B-per": 7, "I-org": 8, "B-gpe": 9, "I-geo": 10, "I-tim": 11, "B-art": 12, "B-eve": 13, "I-art": 14, "I-eve": 15, "I-gpe": 16, "B-nat": 17, "I-nat": 18}'}
```
- 标签的意义
- geo = Geographical entity 
- org = Organization 
- per = Person 
- gpe = Geopolitical entity 
- tim = Time indicator 
- art = Artifact 
- eve = Event 
- nat = Natural phenomenon
- `B-` / `I`  该前缀代表`开始`与`紧跟`例如， `August 19`，`B-tim I-tim`。 这两个词都是时间，所以要表示开始与结束标识，这里的开始就是`Augst` => `B-tim`

每一个字Token化之后，接着用token来表示整个句子，因为计算的过程都是通过数字完成的。

```python
# eval convert string to dictionary
text_vocab = eval(text_config['index_word'])
print("Unique words in vocab:", len(text_vocab))

ner_vocab = eval(ner_config['index_word'])
print("Unique NER tags in vocab:", len(ner_vocab))

# Transforms each text in texts to a sequence of integers.
x_tok = text_tok.texts_to_sequences(data_pd['text'])
y_tok = ner_tok.texts_to_sequences(data_pd['label'])
```
这里打印两个例子，可以看到语义标签以及句子都转成了矩阵（看起来像个数组）
```
# O B-gpe O O O O O O O O O O O O O B-geo O B-tim O O O O O O O
# [2, 9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 4, 2, 2, 2, 2, 2, 2, 2]
print(data_pd['label'][0], y_tok[0])

# An Iraqi court has sentenced 11 men to death for the massive truck bombings in Baghdad last August that killed more than 100 people .
# [316, 89, 233, 13, 1112, 494, 240, 7, 248, 12, 2, 913, 1485, 528, 5, 146, 61, 570, 16, 38, 50, 55, 671, 39, 3]
print(data_pd['text'][0], x_tok[0])
```

把所有输入与输出的数据处理成同样的长度.
因为在tensorflow计算过程中，所有输入输出的的数据需要相同。
过长的数据会被截掉后半段，过段的数据会在句子后加上填充符。
```python
max_len = 50
# padding, String, "pre" or "post" (optional, defaults to "pre"): pad either before or after each sequence.
x_pad = pad_sequences(x_tok, padding='post', maxlen=max_len)
y_pad = pad_sequences(y_tok, padding='post', maxlen=max_len)
print(x_pad.shape, y_pad.shape)
```
最后对输出值进行one-hot 处理。
因为输出结果是个`类别`， 如果用 1 、2、 3 数字描述类别1、类别2、类别3，会得到 类别3 = 类别1 + 类别2 的错误现象。
```
# Since there are multiple labels, each label token needs to be one-hot encoded like so:
num_classes = len(ner_vocab) + 1
Y = to_categorical(y_pad, num_classes)
# (62010, 50, 19)
print(Y.shape)
```
这里打印了一个例子来看看，可以看到B-gpe已经被转化成了长度为19的1维举证。
所以每一句话都是维度[50, 19]的矩阵。
```
# B-gpe => 9 => [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
print('finally covert ', ner_vocab.get('9'), '=>', y_pad[0][1], '=>', Y[0][1])
```

### 构建模型
```python
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

```

### 训练与验证模型
```python
X = x_pad
total_sentences = ner_config.get('document_count')
test_size = round(total_sentences / BATCH_SIZE * 0.2)
test_size = BATCH_SIZE * test_size

X_train = X[test_size:]
Y_train = Y[test_size:]

X_test = X[0:test_size]
Y_test = Y[0:test_size]

model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=15)

model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)
```
准确率为`96.34%`
```
138/138 [==============================] - 2s 4ms/step - loss: 0.0872 - accuracy: 0.9634
```



### 预测句子
```python

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
    template = '|'.join(['{' + str(index) + ': <15}' for index, x in enumerate(x.split(' '))])
    print(template.format(*x.split(' ')))
    print(template.format(*ground_true.split(' ')))
    print(template.format(*prediction.split(' ')))
    print('\n')

```

打印其中两个例子，可以看到预测还是挺准确的
```bash
sentence=>     |An             |Iraqi          |court          |has            |sentenced      |11             |men            |to             |death          |for            |the            |massive        |truck          |bombings       |in             |Baghdad        |last           |August         |that           |killed         |more           |than           |100            |people         |.              |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          
ground_true=>  |O              |B-gpe          |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |B-geo          |O              |B-tim          |O              |O              |O              |O              |O              |O              |O              |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          
prediction=>   |O              |B-gpe          |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |B-geo          |O              |B-tim          |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              



sentence=>     |The            |court          |convicted      |the            |men            |of             |planning       |and            |implementing   |the            |August         |19             |attacks        |on             |the            |Iraqi          |Ministries     |of             |Finance        |and            |Foreign        |Affairs        |.              |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          
ground_true=>  |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |B-tim          |I-tim          |O              |O              |O              |B-gpe          |O              |O              |B-org          |I-org          |I-org          |I-org          |O              |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          |<OOV>          
prediction=>   |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |B-tim          |I-tim          |O              |O              |O              |B-gpe          |B-org          |I-org          |I-org          |I-org          |I-org          |I-org          |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              |O              

```

[bilistm 项目传送门](https://github.com/kk412027247/NER_with_BiLSTM/blob/main/bilistm_2.py)