import os
import datetime

import numpy as np
import tensorflow as tf

from utils import load_data, plot_history_tf, plot_heat_map

# project root path
project_path = "./"
# define log directory
# must be a subdirectory of the directory specified when starting the web application
# it is recommended to use the date time as the subdirectory name
log_dir = project_path + "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = project_path + "jd-1d_ecg_model.h5"

# the ratio of the test set
RATIO = 0.2
# the random seed
RANDOM_SEED = 42
BATCH_SIZE = 128
NUM_EPOCHS = 10


# build the CNN model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(300,)),
        tf.keras.layers.Reshape((300, 1)),

        # Block 1
        tf.keras.layers.Conv1D(32, kernel_size=7, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2),

        # Block 2
        tf.keras.layers.Conv1D(64, kernel_size=5, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2),

        # Block 3
        tf.keras.layers.Conv1D(128, kernel_size=5, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.3),

        # Block 4
        tf.keras.layers.Conv1D(256, kernel_size=3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.3),

        # Global Pooling instead of Flatten
        tf.keras.layers.GlobalAveragePooling1D(),

        # Fully connected layers
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(15, activation='softmax')
    ])
    return model


def main():
    # X_train,y_train is the training set
    # X_test,y_test is the test set
    X_train, X_test, y_train, y_test = load_data(RATIO, RANDOM_SEED)

    np.save("./15_class_signal_train_256_1D_for_all_data_high_8_2.npy", X_train)
    np.save("./15_class_signal_test_256_1D_for_all_data_high_8_2.npy", X_test)
    np.save("./15_class_symbol_train_256_for_all_data_high_8_2.npy", y_train)
    np.save("./15_class_symbol_test_256_for_all_data_high_8_2.npy", y_test)

    
    if os.path.exists(model_path):
        # import the pre-trained model if it exists
        print('Import the pre-trained model, skip the training process')
        model = tf.keras.models.load_model(filepath=model_path)
    else:
        # build the CNN model
        model = build_model()
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        # define the TensorBoard callback object
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        # train and evaluate model
        history = model.fit(X_train, y_train, epochs=NUM_EPOCHS,
                            batch_size=BATCH_SIZE,
                            validation_data=(X_test, y_test),
                            callbacks=[tensorboard_callback])
        # save the model
        model.save(filepath=model_path)
        # plot the training history
        plot_history_tf(history)

    # predict the class of test data
    # y_pred = model.predict_classes(X_test)  # predict_classes has been deprecated
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    # plot confusion matrix heat map
    plot_heat_map(y_test, y_pred)
    

if __name__ == '__main__':
    main()
