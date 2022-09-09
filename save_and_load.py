import os
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import ModelCheckpoint
from keras.metrics import SparseCategoricalAccuracy
from tensorflow import train

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


def create_model():
    model = Sequential([
        Dense(512, activation='relu', input_shape=(784,)),
        Dropout(0.2),
        Dense(10)
    ])
    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[SparseCategoricalAccuracy()])
    return model


# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()

model = create_model()

model.summary()

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Save checkpoints during training

# Create a callback that saves the model's weights
cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

# Train the model with the new callback
# model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels), callbacks=[cp_callback])  # Pass callback to training

loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("accuracy: {:5.2f}%".format(100 * acc))

# build new model without training , its accuracy would be around 10%
model2 = create_model()
loss, acc = model2.evaluate(test_images, test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))
# Untrained model, accuracy: 10.60%


# load wight for the new model
model2.load_weights(checkpoint_path)
loss, acc = model2.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
# Restored model, accuracy: 87.20%

# Checkpoint callback options

checkpoint_path_2 = 'training_2/cp-{epoch:04d}.ckpt'
checkpoint_dir_2 = os.path.dirname(checkpoint_path_2)

batch_size = 32

cp_callback2 = ModelCheckpoint(filepath=checkpoint_path_2, verbose=1, save_weights_only=True, save_freq=5 * batch_size)
# cp_callback = ModelCheckpoint(filepath=checkpoint_path_2, verbose=1, save_weights_only=True, save_best_only=True)
model3 = create_model()

# model3.save_weights(checkpoint_path_2.format(epoch=0))

# model3.fit(train_images, train_labels, epochs=50, batch_size=batch_size, callbacks=[cp_callback2], validation_data=(test_images, test_labels), verbose=0)

latest = train.latest_checkpoint(checkpoint_dir_2)
model4 = create_model()

print(latest)
model4.load_weights(latest)

# Re-evaluate the model
loss, acc = model4.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
# 32/32 - 0s - loss: 0.4787 - sparse_categorical_accuracy: 0.8750 - 95ms/epoch - 3ms/step
# Restored model, accuracy: 87.50%


# Manually save weights

#  after train, save down the weights
model4.save_weights('./checkpoints/my_checkpoint')

model5 = create_model()

# load the weights
model5.load_weights('./checkpoints/my_checkpoint')

# Evaluate the model
loss, acc = model5.evaluate(test_images, test_labels, verbose=2)
print("Restored model5, accuracy: {:5.2f}%".format(100 * acc))
