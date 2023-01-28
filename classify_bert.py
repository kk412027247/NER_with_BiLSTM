import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
from transformers import BertTokenizer, TFBertModel


# using TFDS dataset
# note: as_supervised converts dicts to tuples
imdb_train, ds_info = tfds.load(
    name="imdb_reviews", split="train", with_info=True, as_supervised=True
)
imdb_test = tfds.load(name="imdb_reviews", split="test", as_supervised=True)


# Defining BERT tokenizer
# bert_name = 'bert-base-uncased'
bert_name = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(
    bert_name,
    add_special_tokens=True,
    do_lower_case=False,
    max_length=150,
    pad_to_max_length=True,
)


def bert_encoder(review):
    txt = review.numpy().decode("utf-8")
    encoded = tokenizer.encode_plus(
        txt,
        add_special_tokens=True,
        max_length=150,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        truncation=True,
    )
    return encoded["input_ids"], encoded["token_type_ids"], encoded["attention_mask"]


bert_train = [bert_encoder(r) for r, l in imdb_train]
bert_lbl = [l for r, l in imdb_train]

bert_train = np.array(bert_train)

bert_lbl = tf.keras.utils.to_categorical(bert_lbl, num_classes=2)


# create training and validation splits
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(
    bert_train, bert_lbl, test_size=0.2, random_state=42
)

tr_reviews, tr_segments, tr_masks = np.split(x_train, 3, axis=1)
val_reviews, val_segments, val_masks = np.split(x_val, 3, axis=1)


tr_reviews = tr_reviews.squeeze()
tr_segments = tr_segments.squeeze()
tr_masks = tr_masks.squeeze()

val_reviews = val_reviews.squeeze()
val_segments = val_segments.squeeze()
val_masks = val_masks.squeeze()


def example_to_features(input_ids, attention_masks, token_type_ids, y):
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "token_type_ids": token_type_ids,
    }, y


train_ds = (
    tf.data.Dataset.from_tensor_slices((tr_reviews, tr_masks, tr_segments, y_train))
    .map(example_to_features)
    .shuffle(100)
    .batch(16)
)

valid_ds = (
    tf.data.Dataset.from_tensor_slices((val_reviews, val_masks, val_segments, y_val))
    .map(example_to_features)
    .shuffle(100)
    .batch(16)
)


# prep data for testing
bert_test = [bert_encoder(r) for r, l in imdb_test]
bert_tst_lbl = [l for r, l in imdb_test]

bert_test2 = np.array(bert_test)
bert_tst_lbl2 = tf.keras.utils.to_categorical(bert_tst_lbl, num_classes=2)

ts_reviews, ts_segments, ts_masks = np.split(bert_test2, 3, axis=1)
ts_reviews = ts_reviews.squeeze()
ts_segments = ts_segments.squeeze()
ts_masks = ts_masks.squeeze()


test_ds = (
    tf.data.Dataset.from_tensor_slices(
        (ts_reviews, ts_masks, ts_segments, bert_tst_lbl2)
    )
    .map(example_to_features)
    .shuffle(100)
    .batch(16)
)

bert_name = "bert-base-cased"
bert = TFBertModel.from_pretrained(bert_name)

bert.summary()

max_seq_len = 150
inp_ids = tf.keras.layers.Input((max_seq_len,), dtype=tf.int64, name="input_ids")
att_mask = tf.keras.layers.Input((max_seq_len,), dtype=tf.int64, name="attention_mask")
seg_ids = tf.keras.layers.Input((max_seq_len,), dtype=tf.int64, name="token_type_ids")


inp_dict = {"input_ids": inp_ids, "attention_mask": att_mask, "token_type_ids": seg_ids}
outputs = bert(inp_dict)


x = tf.keras.layers.Dropout(0.2)(outputs[1])
x = tf.keras.layers.Dense(200, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(2, activation="softmax")(x)

custom_model = tf.keras.models.Model(inputs=inp_dict, outputs=x)


# first train the new layers added
bert.trainable = False
optimizer = tf.keras.optimizers.Adam()  # standard learning rate
loss = tf.keras.losses.BinaryCrossentropy()  # from_logits=True)
custom_model.compile(
    optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
)


print("Custom Model: training custom model on IMDB")
custom_history = custom_model.fit(train_ds, epochs=10, validation_data=valid_ds)


custom_model.evaluate(test_ds)


# Now finetune BERT for a couple of epochs
bert.trainable = True
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.BinaryCrossentropy()  # from_logits=True)

custom_model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

custom_model.summary()

print("Custom Model: Fine-tuning BERT on IMDB")
custom_history = custom_model.fit(train_ds, epochs=2, validation_data=valid_ds)

custom_model.evaluate(test_ds)
