import numpy as np
import tensorflow as tf
from keras.applications.resnet_v2 import ResNet50V2
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ResNet50 model

train_path = "Datasets/train"
test_path = "Datasets/test"

# reading dataset labels
train_labels = pd.read_csv("Datasets/labels.csv")
test_labels = pd.read_csv("Datasets/sample_submission.csv")

train_labels["id"] = train_labels["id"].apply(lambda x: x + ".jpg")
test_labels["id"] = test_labels["id"].apply(lambda x: x + ".jpg")


print("shape", np.shape(train_labels))
unique = train_labels.breed.unique()
classes = [
    "african_hunting_dog",
    "dingo",
    "bluetick",
    "cairn",
    "toy_poodle",
    "norwegian_elkhound",
    "walker_hound",
    "groenendael",
    "affenpinscher",
    "basset",
]
for i in unique:
    if i not in classes:
        train_labels = train_labels.drop(train_labels[train_labels["breed"] == i].index)


plot = plt.figure(figsize=(18, 12))
grph = sns.countplot(x="breed", data=train_labels)
plt.xlabel("Breeds", fontsize=22)
plt.ylabel("Number of pictures", fontsize=22)
plt.title("Selected Breeds", fontsize=26)
plt.xticks(fontsize=16, rotation=15)
plt.yticks(fontsize=16)
plt.savefig("resnet50v2/selected_breeds_2.png")
plt.clf()
plt.cla()

# Set up the image data generator for training data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=20,
    brightness_range=[0.2, 1.0],
)

# Define the training set with 16 images per batch and 'categorical' class mode.
# The images are resized to 224x224 and shuffled.
train_set = train_datagen.flow_from_dataframe(
    dataframe=train_labels,
    directory=train_path,
    x_col="id",
    y_col="breed",
    batch_size=16,
    subset="training",
    class_mode="categorical",
    target_size=(224, 224),
    seed=42,
    shuffle=True,
)


# Define the validation set with 224x224 image size, 16 batch size, 'categorical' class mode and 'validation' subset.
validate_set = train_datagen.flow_from_dataframe(
    dataframe=train_labels,
    directory=train_path,
    x_col="id",
    y_col="breed",
    batch_size=16,
    subset="validation",
    class_mode="categorical",
    target_size=(224, 224),
    seed=42,
    shuffle=True,
)


# only rescaling is applied
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Define the test set using a test data generator
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_set = test_datagen.flow_from_dataframe(
    dataframe=test_labels,
    directory=test_path,
    x_col="id",
    y_col=None,
    batch_size=16,
    class_mode=None,
    seed=42,
    shuffle=False,
    target_size=(224, 224),
)

# Load pre-trained ResNet50V2 model
resnet = ResNet50V2(input_shape=[224, 224, 3], weights="imagenet", include_top=False)

# Freeze weights to avoid re-training
for layer in resnet.layers:
    layer.trainable = False

# Flatten output and add dropout layer
x = keras.layers.Flatten()(resnet.output)
x = keras.layers.Dropout(0.4)(x)

# Add final output layer with 10 neurons
output = keras.layers.Dense(10, activation="softmax")(x)

# Create the model and compile it
model = tf.keras.models.Model(inputs=resnet.input, outputs=output)
# Defining optimiser
opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

# Define number of train and validation steps
train_steps = train_set.n // train_set.batch_size
validate_steps = validate_set.n // validate_set.batch_size

# Train the model
resnet50 = model.fit(
    train_set,
    validation_data=validate_set,
    epochs=30,
    steps_per_epoch=train_steps,
    validation_steps=validate_steps,
)

# Evaluate the model on validation set
val_loss, val_acc = model.evaluate(validate_set)
print(
    "Validation loss: {:.2f}, Validation accuracy: {:.2f}%".format(
        val_loss, val_acc * 100
    )
)

# Save the model
model.save("resnet_model/model_1")

# Load the saved model
# model = keras.models.load_model('resnet_model/model_1')

# Make predictions on test set and calculate accuracy
y_pred = model.predict(test_set)
predicted_labels = np.argmax(y_pred, axis=1)
true_labels = np.argmax(test_labels, axis=1)
accuracy = np.mean(predicted_labels == true_labels)
print("Test accuracy: {:.2f}%".format(accuracy * 100))

# Plot a confusion matrix
confusion_matrix = pd.crosstab(
    true_labels, predicted_labels, rownames=["Correct Labels"], colnames=["Predicted"]
)
sns.heatmap(confusion_matrix, annot=False, fmt="g")
plt.show()
# predicting output
STEP_SIZE_TEST = test_set.n // test_set.batch_size

# if we don't reset we'll get output in an unordered complex manner
test_set.reset()

pred = model.predict(test_set, steps=STEP_SIZE_TEST, verbose=1)

plt.plot(resnet50.history["accuracy"])
plt.plot(resnet50.history["val_accuracy"])
plt.title("Model accuracy", fontsize=34)
plt.ylabel("Accuracy", fontsize=30)
plt.xlabel("Epoch", fontsize=30)
plt.legend(["Train", "Test"], loc="upper left", fontsize=30)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig("resnet50v2/accuracy_validation_3.png")
plt.show()
plt.clf()
plt.cla()

plt.plot(resnet50.history["loss"])
plt.plot(resnet50.history["val_loss"])
plt.title("Model loss", fontsize=34)
plt.ylabel("Loss", fontsize=30)
plt.xlabel("Epoch", fontsize=30)
plt.legend(["Train", "Test"], loc="upper right", fontsize=30)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig("resnet50v2/loss_3.png")
plt.show()
plt.clf()
plt.cla()


# create a 2D matrix of accuracy values
acc_matrix = []
for i in range(len(resnet50.history["accuracy"])):
    row = []
    for j in range(len(resnet50.history["val_accuracy"])):
        row.append(resnet50.history["accuracy"][i])
    acc_matrix.append(row)

# plot the heatmap
plt.figure(figsize=(10, 10))
plt.imshow(acc_matrix, cmap="hot", interpolation="nearest")
plt.title("Accuracy Heatmap", fontsize=34)
plt.ylabel("Epoch", fontsize=30)
plt.xlabel("Epoch", fontsize=30)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

# add colorbar
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=24)

# show the plot
plt.savefig("resnet50v2/accuracy_heatmap_1.png")
plt.show()
plt.clf()
plt.cla()
