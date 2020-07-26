import tensorflow as tf
from draw import draw_results

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

clothes = ["T-shirt/top",
           "Trouser",
           "Pullover",
           "Dress",
           "Coat",
           "Sandal",
           "Shirt",
           "Sneaker",
           "Bag",
           "Ankel boot"]

normalize_number = 255.0
train_images, test_images = train_images / \
    normalize_number, test_images / normalize_number

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

loss, accuracy = model.evaluate(test_images, test_labels)

predictions = model.predict(test_images)

draw_results(test_images, test_labels, predictions, clothes)
