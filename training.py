import json
import os
from tensorflow import keras
import matplotlib.pyplot as plt

TRAINING_SET_ADDRESS = 'D:\\Desarrollo\\tesis\\app\\sets\\training_set'
TESTING_SET_ADDRESS = 'D:\\Desarrollo\\tesis\\app\\sets\\testing_set'

training_set_items = []
training_set_labels = []

testing_set_items = []
testing_set_labels = []


def load_data():
    for filename in os.listdir(f"{TRAINING_SET_ADDRESS}\\"):
        with open(f"{TRAINING_SET_ADDRESS}\\{filename}") as file:
            json_info = json.load(file)
            for item in json_info:
                item_info = [int(item['pre_RR']),
                             int(item['post_RR']),
                             int(item['local_RR_average']),
                             int(item['global_RR_average']),
                             ]
                beat_samples = [float(sample) for sample in item['samples']]
                item_info.extend(beat_samples)

                training_set_items.append(item_info)

                item_class = (0 if item['class'] == 'N' else 1)
                item_info.append(item_class)

                training_set_labels.append(item_class)

    for filename in os.listdir(f"{TESTING_SET_ADDRESS}\\"):
        with open(f"{TESTING_SET_ADDRESS}\\{filename}") as file:
            json_info = json.load(file)
            for item in json_info:
                item_info = [int(item['pre_RR']),
                             int(item['post_RR']),
                             int(item['local_RR_average']),
                             int(item['global_RR_average']),
                             ]
                beat_samples = [float(sample) for sample in item['samples']]
                item_info.extend(beat_samples)

                testing_set_items.append(item_info)

                item_class = (0 if item['class'] == 'N' else 1)
                item_info.append(item_class)

                testing_set_labels.append(item_class)


def create_model():
    model = keras.Sequential([keras.layers.Dense(units=5, input_shape=(54,), activation='relu'),
                              keras.layers.Dense(units=10, activation='relu'),
                              keras.layers.Dense(units=30, activation='relu'),
                              keras.layers.Dense(units=50, activation='relu'),
                              keras.layers.Dense(units=30, activation='relu'),
                              keras.layers.Dense(units=10, activation='relu')])

    # output layer
    n_output_clases = 2
    model.add(keras.layers.Dense(units=n_output_clases, activation='softmax'))

    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    return model


load_data()
model = create_model()

print(f"Training set items size: {len(training_set_items)}")
print(f"Training set labels size: {len(training_set_labels)}")

print(f"Testing set items size: {len(testing_set_items)}")
print(f"Testing set labels size: {len(testing_set_labels)}")

history = model.fit(training_set_items, training_set_labels, epochs=50)

# # test
# test_loss, test_acc = model.evaluate(testing_set_items,  testing_set_labels, verbose=2)
#
# print('\nTest accuracy:', test_acc)
#
# training_loss = history.history['loss']
#
# # Create count of the number of epochs
# epoch_count = range(1, len(training_loss) + 1)
#
# # Visualize loss history
# plt.plot(epoch_count, training_loss, 'r--')
# plt.legend(['Training Loss'])
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show();
# min(training_loss)
