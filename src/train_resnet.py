#import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks # type: ignore
from tensorflow.keras.preprocessing import image_dataset_from_directory # type: ignore
from tensorflow import losses, image, data
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions # type: ignore


import pathlib





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

#def change_inputs(images, labels):
#    images = image.resize(images,
#                   [224, 224],
#                   method=image.ResizeMethod.NEAREST_NEIGHBOR )
#    return images, labels

batch_size = 32
img_height = 224
img_width  = 224

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




es = callbacks.EarlyStopping(patience=10, restore_best_weights=True)



#train_ds = train_ds.map(change_inputs)
#val_ds   =   val_ds.map(change_inputs)

# Improve performance with prefetching
train_ds = train_ds.prefetch(data.AUTOTUNE)
val_ds   = val_ds.prefetch(data.AUTOTUNE)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model
# Add custom layers for your classification task

x = layers.Rescaling(1./255)(base_model.output)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
output = layers.Dense(8, activation='softmax')(x)  # Change 8 to the number of your classes

model = models.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer= optimizers.Adam(),
    loss=losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs = 1000,
    callbacks = [es],
    verbose = 1
)

ax1, ax2 = plot_history(history, exp_name= 'First Resnet model')
plt.show()
