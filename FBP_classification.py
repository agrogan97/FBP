'''
Summary:
        - FBP classification using keras
'''

# Read in data
# Normalise images
# Format in train/test/val and do the same for labels
# Build network
# compile

# Images are found in SCUT-FBP5500_v2\Images
# Labels are All_Ratings.xslx

import pandas as pd
from PIL import Image
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.models import model_from_json

def img_to_pixels(img_link):
    '''
    Summary:
            - Take an img and return the pixels

    Returns:
            - pixels, a numpy array of shape 350*350*3
                NB this has already been flattened!
    '''

    im = Image.open(str(img_link))
    pix = im.load()

    pixels_list = []

    # Get as numpy array of size 350*350*3
    for i in range(im.size[0]):
        for j in range(im.size[1]):
            pixels_list.append(pix[i, j])

    pix_arr = np.asarray(pixels_list)

    pix_new_arr = np.reshape(pix_arr, (350,350,3))

    return pix_new_arr

def network():
    '''
    Summary:
            - Define the model
    '''

    visible = Input(shape=(350,350,3))
    conv1 = Conv2D(32, kernel_size=4, activation='relu')(visible)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(16, kernel_size=4, activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    flat = Flatten()(pool2)
    hidden1 = Dense(10, activation='relu')(flat)
    output = Dense(5, activation='softmax')(hidden1)

    model = Model(inputs=visible, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer="SGD", metrics=['accuracy'])
    model.summary()

    return model

def fit_model(model, x_train, y_train, x_val, y_val, epochs, batch_size):
    '''
    Train model
    '''

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size)

    # serialize model to JSON
    model_json = model.to_json()
    with open("models/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("models/model.h5")
    print("Saved model to disk")

    return history

def one_hot_labels(labels):
    '''
    One hot-encode the labels
    '''

    # There are 5:

    label_encoding = {
                        '1' : [1, 0, 0, 0, 0],
                        '2' : [0, 1, 0, 0, 0],
                        '3' : [0, 0, 1, 0, 0],
                        '4' : [0, 0, 0, 1, 0],
                        '5' : [0, 0, 0, 0, 1],
    }

    o_h_labels = []

    for label in labels:
        tmp = label_encoding[str(label)]
        o_h_labels.append(tmp)

    o_h_labels_arr = np.asarray(o_h_labels)

    return o_h_labels_arr

def load_model():
    '''
    Load model and weights from disk
    '''
    # load json and create model
    json_file = open('models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("models/model.h5")
    print("Loaded model from disk")

    return loaded_model

def prep_data(factor, generation_size):
    data = []

    df = pd.read_csv('SCUT-FBP5500_v2/All_Ratings.csv')
    # Get all the img values
    filenames = df['Filename'].to_list()
    ratings = df['Rating'].to_numpy()
    total_num = generation_size
    start = factor * total_num
    num = 0
    print("Beginning data readin ...")
    for img in filenames:
        if start <= num < (start+total_num):
            pixels = img_to_pixels('SCUT-FBP5500_v2/Images/' + str(img))
            data.append(pixels)
        num = num + 1

    pixels_arr = np.asarray(data)
    print("Shape of x_data:", pixels_arr.shape)

    # Get labels:
    labels = np.asarray(ratings)
    labels = labels[start:(start+total_num)]

    oh_labels = one_hot_labels(labels) # Alternatively use np_utils.to_categorical?

    print("Shape of labels:", oh_labels.shape)

    return pixels_arr, oh_labels


def main():

    PARAMS = {
                'epochs' : 100,
                'batch_size' : 32,
                'load_model' : False,
                'generations' : 2,
                'generation_size' : 1000,
    }

    if not PARAMS['load_model']:
        print("Generation %d" % (0) )
        pixels_arr, oh_labels = prep_data(0, PARAMS['generation_size'])
        # Define model
        model = network()

        x_data = pixels_arr / 255.0
        x_train, x_val, y_train, y_val = train_test_split(x_data, oh_labels, test_size=0.2)

        history = fit_model(model, x_train, y_train, x_val, y_val, PARAMS['epochs'], PARAMS['batch_size'])

        PARAMS['load_model'] = True
    else:
        print("Skipping init run")

    generation_num = 1
    while (generation_num < PARAMS['generations']) and PARAMS['load_model']:
        print("Generation %d" % (generation_num) )
        pixels_arr, oh_labels = prep_data(generation_num, PARAMS['generation_size'])

        print("Loading model ...")
        model = load_model()
        print(" ... Done")
        model.compile(loss='categorical_crossentropy', optimizer="SGD", metrics=['accuracy'])

        # Don't forget to normalise the images
        x_data = pixels_arr / 255.0
        x_train, x_val, y_train, y_val = train_test_split(x_data, oh_labels, test_size=0.2)

        history = fit_model(model, x_train, y_train, x_val, y_val, PARAMS['epochs'], PARAMS['batch_size'])

        generation_num = generation_num + 1

if __name__ == "__main__" : main()
