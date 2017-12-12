import csv
import cv2
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout
from sklearn.model_selection import train_test_split


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images, measurements = [], []
            for batch_sample in batch_samples:
                for i in range(3):
                    file_path = './data/IMG/' + batch_sample[i].split('/')[-1]
                    image = cv2.imread(file_path)
                    images.append(image)
                measurements.append(float(batch_sample[3]))
                measurements.append(float(batch_sample[3]) + 0.30)
                measurements.append(float(batch_sample[3]) - 0.30)
            # Data augment
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement * -1.0)
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)


def createmodel():
    # Define model
    model = Sequential()
    # Normalization
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    # Cropping
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Dropout(0.5))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


def main():
    # Prepare data
    lines = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)
    model = createmodel()
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(
        train_generator,
        samples_per_epoch=len(train_samples),
        validation_data=validation_generator,
        nb_val_samples=len(validation_samples), nb_epoch=20
    )

    model.save('model.h5')

if __name__ == "__main__":
    main()
