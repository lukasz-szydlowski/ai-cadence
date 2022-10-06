import os
import requests as req
import json as js
import pandas as pd
import os.path as path
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import tensorflow as tf


# Main function
def main():
    downloadData()
    readAndProcessDataFiles()
    plotSpectrogram()
    model = loadModel()
    x_train, x_val, y_train, y_val = getDataSets()
    fitModel(model, x_train, x_val, y_train, y_val)
    evaluateModel(model, x_val, y_val)


# Downloads data set from key-value cloud based bucket and saves it to local folder
def downloadData():
    url = "https://kvdb.io/V6Apyn8NgYB9fE2p1eH6AA/"

    file_name = path.join("raw_data", "keys.json")

    if path.exists(file_name):
        return

    # getting storage keys
    response = req.get("{}keys".format(url))
    data = js.loads(response.text)

    with open(file_name, "w+") as file:
        file.write(response.text)

    # getting values
    for key in data["keys"]:
        response = req.get("{}{}".format(url, key))
        file_name = path.join("raw_data", "{}.json".format(key))

        if path.exists(file_name):
            continue

        with open(file_name, "w+") as file:
            file.write(response.text)


# Reads and processes all data files
def readAndProcessDataFiles():
    processed_data_file_name = path.join("processed_data", "data.json")

    if path.exists(processed_data_file_name):
        return

    files = [file for file in os.listdir("raw_data") if path.isfile(path.join("raw_data", file)) and file != "keys.json"]
    result = pd.DataFrame()
    index = 0

    for file_name in files:
        file_name = path.join("raw_data", file_name)
        data = pd.read_json(file_name)

        x = np.array([])
        y = np.array([])
        z = np.array([])
        cadence = np.array([])

        # flatting pandas object to array
        for column in data.columns:
            sequence = data[column]

            x = np.append(x, sequence["X"])
            y = np.append(y, sequence["Y"])
            z = np.append(z, sequence["Z"])
            cadence = np.append(cadence, sequence["Cadence"])

        interval = int(5)
        n_splits = int(60 / interval)

        # splitting 1 minute long data sample into 12 samples, 5 seconds long each
        x_splits = np.array_split(x, n_splits)
        y_splits = np.array_split(y, n_splits)
        z_splits = np.array_split(z, n_splits)
        cadence_splits = np.array_split(cadence, n_splits)

        for n in range(n_splits):
            x_split = np.zeros((25, 10))
            y_split = np.zeros((25, 10))
            z_split = np.zeros((25, 10))
            cadence_split = cadence_splits[n]

            # filter out all samples where cadence was 0 (very noisy data)
            if any([c == 0 for c in cadence_split]):
                continue

            # splitting 5 seconds long data samples into 25 samples, 0.2 second long each to compute fft
            x_fft_splits = np.split(x_splits[n], 25)
            y_fft_splits = np.split(y_splits[n], 25)
            z_fft_splits = np.split(z_splits[n], 25)

            for fft_index in range(25):
                x_fft = fft.fft(x_fft_splits[fft_index])
                y_fft = fft.fft(y_fft_splits[fft_index])
                z_fft = fft.fft(z_fft_splits[fft_index])

                x_split[fft_index] = np.append(np.abs(x_fft), np.angle(x_fft))
                y_split[fft_index] = np.append(np.abs(y_fft), np.angle(y_fft))
                z_split[fft_index] = np.append(np.abs(z_fft), np.angle(z_fft))

            result = pd.concat([result, pd.DataFrame({
                "Index": index,
                "X": [x_split],
                "Y": [y_split],
                "Z": [z_split],
                "Cadence": [cadence_split]})])

            index = index + 1

    result.set_index("Index", drop=False, inplace=True)
    pd.json_normalize(result)
    json_data = result.to_json()

    with open(processed_data_file_name, "w+") as file:
        file.write(json_data)


# Plots spectrogram of given 1 minute long data sample for X axis
def plotSpectrogram():
    file_name = "2022_09_07_11_59_01_03420.json"
    # file_name = "2022_09_07_12_22_00_04320.json"
    # file_name = "2022_09_05_11_41_13_00060.json"
    data = pd.read_json(path.join("raw_data", file_name))

    x = np.array([])
    cadence = np.array([])

    for column in data.columns:
        sequence = data[column]
        x = np.append(x, sequence["X"])
        cadence = np.append(cadence, sequence["Cadence"])

    samples = 25
    ffts = np.zeros((1500 - samples, samples))

    for i in range(1500 - samples):
        xs = x[i:i + samples]
        ffts[i] = np.abs(fft.fft(xs))

    fig = plt.figure(figsize=(18, 6))
    pl = fig.add_subplot()

    plt_x, plt_y = np.meshgrid(range(1500 - samples), range(samples))

    pl.pcolor(plt_x, plt_y, ffts.T)
    plt.xticks(range(0, 1500, samples), cadence, rotation='vertical')

    fig.canvas.manager.set_window_title('Example spectrogram')
    plt.xlabel('Cadence in time (60 sec)')
    plt.ylabel('Amplitude')

    plt.savefig(path.join("images", "example spectrogram.jpg"))
    plt.show()


# Loading model
def loadModel():
    model_name = path.join("models", "model1")

    try:
        model = tf.keras.models.load_model(model_name)
        model.summary()

        return model
    except:
        pass

    dropout = 0.1

    x_input = tf.keras.Input(shape=[25, 10], name="input_1")
    y_input = tf.keras.Input(shape=[25, 10], name="input_2")
    z_input = tf.keras.Input(shape=[25, 10], name="input_3")

    x = tf.keras.layers.Concatenate()([x_input, y_input, z_input])

    x = tf.keras.layers.Conv1D(64, kernel_size=5, strides=5, name="conv_1")(x)
    x = tf.keras.layers.Dropout(rate=dropout, name="dropout_1")(x)

    x = tf.keras.layers.BatchNormalization(name="batch_normalization_1")(x)
    x = tf.keras.layers.Activation('sigmoid', name="activation_1")(x)
    x = tf.keras.layers.Dropout(rate=dropout, name="dropout_5")(x)

    x = tf.keras.layers.SimpleRNN(32, return_sequences=True, name="gru_1")(x)
    x = tf.keras.layers.Dropout(rate=dropout, name="dropout_6")(x)
    x = tf.keras.layers.BatchNormalization(name="batch_normalization_2")(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation="relu", name="dense_3"),
                                        name="time_distributed")(x)

    model = tf.keras.Model(inputs=[x_input, y_input, z_input], outputs=x)

    model.save(model_name)
    model.summary()

    return model


# Obtaining train and validation data sets based of processed data (0.8 / 0.2 split)
def getDataSets():
    processed_data_file_name = path.join("processed_data", "data.json")

    data = pd.read_json(processed_data_file_name)
    data.set_index("Index", drop=True, inplace=True)
    pd.json_normalize(data)

    array = data.to_numpy()
    n = array.shape[0]
    x = np.zeros([n, 3, 25, 10])
    y = np.zeros([n, 5])

    for i in range(n):
        x[i, 0] = array[i, 0]
        x[i, 1] = array[i, 1]
        x[i, 2] = array[i, 2]
        y[i] = array[i, 3]

    x_train, x_val, y_train, y_val = ms.train_test_split(x, y, train_size=0.8, random_state=2022, shuffle=True)

    return x_train, x_val, y_train, y_val


# Fitting the model to the training set
def fitModel(model, x_train, x_val, y_train, y_val):
    checkpoint_name = path.join("models", "model1", "cp4.ckpt")

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=200, restore_best_weights=True,
                                                      verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_name, save_weights_only=True, save_freq=1000,
                                                    verbose=1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer, loss='mse', metrics=["mae"])

    try:
        model.load_weights(checkpoint_name)
        return
    except:
        pass

    model.fit(x=[x_train[:, 0], x_train[:, 1], x_train[:, 2]], y=y_train,
              validation_data=[[x_val[:, 0], x_val[:, 1], x_val[:, 2]], y_val],
              batch_size=16, epochs=10000, callbacks=[early_stopping, checkpoint])

    model.save_weights(checkpoint_name)


# Evaluating model using validation set
def evaluateModel(model, x_val, y_val):
    model.evaluate(x=[x_val[:, 0], x_val[:, 1], x_val[:, 2]], y=y_val)

    predictions = model.predict(x=[x_val[:, 0], x_val[:, 1], x_val[:, 2]])

    fig = plt.figure(figsize=(18, 6))
    pl = plt.subplot()
    pl.hist(y_val.flatten(), bins=100, density=True)

    fig.canvas.manager.set_window_title('Cadence histogram - Validation set labels')
    plt.xlabel('Cadence')
    plt.ylabel('Density')

    plt.savefig(path.join("images", "histogram - labels.jpg"))
    plt.show()

    fig = plt.figure(figsize=(18, 6))
    pl = plt.subplot()
    pl.hist(predictions.flatten(), bins=100, density=True)

    fig.canvas.manager.set_window_title('Cadence histogram - Validation set predictions')
    plt.xlabel('Cadence')
    plt.ylabel('Density')

    plt.savefig(path.join("images", "histogram - predictions.jpg"))
    plt.show()

    yp_list = list(zip(y_val.flatten(), predictions.flatten()))
    cadences = np.arange(1, 115, 1)
    avg_errors = np.zeros(cadences.shape)

    for index, cadence in enumerate(cadences):
        avg_errors[index] = 100 * np.average([abs(yp[1] - cadence) for yp in yp_list if yp[0] == cadence]) / cadence

    fig = plt.figure(figsize=(18, 6))
    pl = plt.subplot()
    plt.plot(cadences, avg_errors, 'r')

    fig.canvas.manager.set_window_title('Average absolute errors in % - Predictions on validation set')
    plt.xlabel('Cadence')
    plt.ylabel('Absolute error (%)')

    plt.savefig(path.join("images", "average error.jpg"))
    plt.show()


# Program entry point
if __name__ == '__main__':
    main()
