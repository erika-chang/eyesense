#import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks # type: ignore
from tensorflow.keras.preprocessing import image_dataset_from_directory # type: ignore
from tensorflow import losses
import matplotlib.pyplot as plt


import pathlib


def initialize_model():

    model = models.Sequential([
           layers.Rescaling(1./255),
           layers.Conv2D(32, 3, activation='relu'),
           layers.MaxPooling2D(pool_size=2),
           #layers.Dropout(0.1),
           layers.Conv2D(64, 3, activation='relu'),
           layers.MaxPooling2D(pool_size=2),
           #layers.Dropout(0.2),
           layers.Conv2D(128, 3, activation='relu'),
           layers.MaxPooling2D(pool_size=2),
           layers.Dropout(0.6),
           layers.Flatten(),
           layers.Dense(128, activation='relu'),
           layers.Dense(8, activation='softmax')
        ])

    optimizer = optimizers.Adam(learning_rate=0.002)
    ### Model compilation
    model.compile(
          optimizer=optimizer,
          loss = losses.SparseCategoricalCrossentropy(from_logits=False),
          metrics=['accuracy'])

    return model



def plot_history(history, title='', axs=None, exp_name=""):
    if axs is not None:
        ax1, ax2 = axs
    else:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if len(exp_name) > 0 and exp_name[0] != '_':
        exp_name = '_' + exp_name
    ax1.plot(history.history['loss'], label = 'train' + exp_name)
    ax1.plot(history.history['val_loss'], label = 'val' + exp_name)
    #ax1.set_ylim(0., 2.2)
    ax1.autoscale()
    ax1.set_title('loss')
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='train accuracy'  + exp_name)
    ax2.plot(history.history['val_accuracy'], label='val accuracy'  + exp_name)
    #ax2.set_ylim(0.25, 1.)
    ax2.autoscale()
    ax2.set_title('Accuracy')
    ax2.legend()
    return (ax1, ax2)

'''
#We can add a Rescaling layer into the model
def change_inputs(images, labels):
    x = image.resize(normalization_layer(images),
                   [256, 256],
                   method=image.ResizeMethod.NEAREST_NEIGHBOR )
    return x, x
'''

batch_size = 32
img_height = 256
img_width  = 256

#Adjust the path according to your machine
data_dir_train = pathlib.Path('../my_notebooks/data_train/')

train_ds = image_dataset_from_directory(
            data_dir_train,
            validation_split=0.2,
            subset="training",
            seed=10,
            image_size=(img_height, img_width),
            batch_size=batch_size)

val_ds = image_dataset_from_directory(
            data_dir_train,
            validation_split=0.2,
            subset="validation",
            seed=10,
            image_size=(img_height, img_width),
            batch_size=batch_size)


for image_batch, labels_batch in train_ds:
  print(f"👉The shape of each train batch is {image_batch.shape}")
  print(f"  The shape of each target batch is {labels_batch.shape}")
  break


#tf.keras.backend.clear_session()
model = initialize_model()

es = callbacks.EarlyStopping(patience=10, restore_best_weights=False)

#normalization_layer = layers.Rescaling(1./255)

#normalized_ds = train_ds.map(change_inputs)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs = 1000,
    callbacks = [es],
    verbose = 1
)

ax1, ax2 = plot_history(history, exp_name= 'First CNN model')
plt.show()
