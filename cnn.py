import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Flatten, Convolution2D
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm

# CNN
model_saved = False

# Defining paths
train_path = "Datasets/train"
test_path = "Datasets/test"

# Reading dataset labels
train_labels = pd.read_csv("Datasets/labels.csv")
test_labels = pd.read_csv("Datasets/sample_submission.csv")

# Some files are jpeg, changing all of them to be .jpg
train_labels["id"] = train_labels["id"].apply(lambda x: x + ".jpg")
test_labels["id"] = test_labels["id"].apply(lambda x: x + ".jpg")

lables = pd.read_csv("Datasets/labels.csv")
breed_count = lables["breed"].value_counts()

targets = pd.Series(lables["breed"])
one_hot = pd.get_dummies(targets, sparse=True)
one_hot_labels = np.asarray(one_hot)

img_rows = 128
img_cols = 128
num_channel = 1  # 3 colour channes

img_1 = cv2.imread("Datasets/train/0cf4dabd83d91e22f6ce845fe81fa21d.jpg", 0)
cv2.imwrite("cnn/example_image_0.png", img_1)
img_2 = cv2.imread("Datasets/train/0ea78b024dc3955332d3ddb08b8e50f0.jpg", 0)
cv2.imwrite("cnn/example_image_1.png", img_2)
img_3 = cv2.imread("Datasets/train/0d103ca7cf575757374f8f6ae87d8868.jpg", 0)
cv2.imwrite("cnn/example_image_2.png", img_3)
img_4 = cv2.imread("Datasets/train/1d7b95ca93d943e74adb6f6777d3ac6e.jpg", 0)
cv2.imwrite("cnn/example_image_3.png", img_4)
img_5 = cv2.imread("Datasets/train/1dc08faffb5d49f608ebffe5f293e67b.jpg", 0)
cv2.imwrite("cnn/example_image_4.png", img_5)

x_feature = []
y_feature = []

i = 0  # initialisation
for f, img in tqdm(lables.values):  # f for format ,jpg
    train_img = cv2.imread("Datasets/train/{}.jpg".format(f), 0)
    label = one_hot_labels[i]
    train_img_resize = cv2.resize(train_img, (img_rows, img_cols))
    x_feature.append(train_img_resize)
    y_feature.append(label)
    i += 1

x_train_data = np.array(x_feature, np.float32) / 255.0  # /= 255 for normolisation
x_train_data = np.expand_dims(x_train_data, axis=3)

y_train_data = np.array(y_feature, np.uint8)

# Splitting data into 80/20 ratio
x_train, x_val, y_train, y_val = train_test_split(
    x_train_data, y_train_data, test_size=0.2, random_state=2
)
# Splitting train data into 90/10
x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, test_size=0.1, random_state=2
)

if model_saved == False:
    # Define a Sequential model
    model = Sequential()

    # Add a convolutional layer with 64 filters and a 4x4 kernel, followed by ReLU activation and max pooling
    model.add(
        Convolution2D(
            filters=64,
            kernel_size=(4, 4),
            padding="Same",
            activation="relu",
            input_shape=(img_rows, img_cols, num_channel),
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Add another convolutional layer with 64 filters and a 4x4 kernel, followed by ReLU activation and max pooling
    model.add(
        Convolution2D(filters=64, kernel_size=(4, 4), padding="Same", activation="relu")
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the output of the convolutional layers
    model.add(Flatten())

    # Add a fully connected layer with 120 units and ReLU activation
    model.add(Dense(units=120, activation="relu"))

    # Add the output layer with 120 units and softmax activation
    model.add(Dense(units=120, activation="softmax"))

    # Compile the model with Adam optimizer, categorical crossentropy loss, and accuracy metric
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Print the summary of the model
    model.summary()

    # Save the model to a file
    model.save("cnn_model/model_1_10epoch")

    # Train the model with a batch size of 128 and 30 epochs, using the training and validation data
    batch_size = 128
    nb_epochs = 30
    cnn = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=nb_epochs,
        verbose=2,
        validation_data=(x_val, y_val),
        initial_epoch=0,
    )


# model = keras.models.load_model('cnn_model\model_1_10epoch')

# Get the predicted labels
y_pred = model.predict(x_test)
predicted_labels = np.argmax(y_pred, axis=1)

# Get the true labels
true_labels = np.argmax(y_test, axis=1)

# Calculate the accuracy
accuracy = np.mean(predicted_labels == true_labels)

print("Accuracy: {:.2f}%".format(accuracy * 100))

# Plot a confusion matrix
confusion_matrix = pd.crosstab(
    true_labels, predicted_labels, rownames=["Correct Labels"], colnames=["Predicted"]
)
sns.heatmap(confusion_matrix, annot=False, fmt="g")
plt.show()

# Plotting accuracy validation
plt.plot(cnn.history["accuracy"])
plt.plot(cnn.history["val_accuracy"])
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper left")
plt.savefig("cnn/accuracy_validation_4.png")
plt.show()
plt.clf()
plt.cla()


# Plotting model loss
plt.plot(cnn.history["loss"])
plt.plot(cnn.history["val_loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper left")
plt.savefig("cnn/loss_4.png")
plt.show()
plt.clf()
plt.cla()

# create a 2D matrix of accuracy values
acc_matrix = []
for i in range(len(cnn.history["accuracy"])):
    row = []
    for j in range(len(cnn.history["val_accuracy"])):
        row.append(cnn.history["accuracy"][i])
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
plt.savefig("cnn/accuracy_heatmap_04.png")
plt.show()
plt.clf()
plt.cla()
