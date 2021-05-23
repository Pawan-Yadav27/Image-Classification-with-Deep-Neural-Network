
# Loading Fashion MNIST dataset
import tensorflow as tf
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


X_train.shape, X_test.shape , y_train.shape , y_test.shape


class_names = ["T-shirt / top", "Trouser", "Pullover", "Dress",
        "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


import matplotlib.pyplot as plt
plt.figure()
plt.imshow (X_train[1])
plt.colorbar()
plt.grid(False)
plt.show()


X_train = X_train / 255.0
X_test = X_test / 255.0



model = tf.keras.Sequential ([
   tf.keras.layers.Flatten (input_shape = (28, 28), name = "Input"),
   tf.keras.layers.Dense (256, activation = 'relu', name = "Hidden"),
   tf.keras.layers.Dense (10, name = "Output")
])



model.summary()

model.layers

hidden = model.layers[1]
print(hidden.name)


weights, biases = hidden.get_weights ()
print(weights)

print(biases)

weights.shape, biases.shape


model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(
                         from_logits = True),
              optimizer = 'adam',
              metrics = ['accuracy'])

# history = model.fit (X_train, y_train, epochs = 10, validation_data= (X_test , y_test))
                     # validation_split = 0.3)
import pandas as pd
import numpy as np
X_train = np.vstack((X_train , X_test))
y_train = np.hstack((y_train , y_test))
history = model.fit (X_train, y_train, epochs = 20, validation_data= (X_test , y_test))

import pandas as pd
pd.DataFrame (history.history).plot (figsize = (8, 5))
plt.grid(True)
plt.show()


test_loss, test_acc = model.evaluate (X_test, y_test, verbose = 2)
print ('\ nTest accuracy:', test_acc)



probability_model = tf.keras.Sequential ([model, tf.keras.layers.Softmax()])


predictions = probability_model.predict(X_test)

predictions[0]

import numpy as np
np.argmax(predictions[0])


y_test[0]
