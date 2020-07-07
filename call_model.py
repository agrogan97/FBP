import numpy as np
import FBP_classification as fbp
from PIL import Image
import matplotlib.pyplot as plt


def format_inputs(filename):
    '''
    Need a 350x350 res image, so change the resolution
    '''

    test_img_dir = 'test_cases/' + str(filename)

    im = Image.open(test_img_dir)
    size = 350,350
    im_resized = im.resize(size, Image.ANTIALIAS)
    new_dir_name = test_img_dir[:-4] + '_newres.png'
    im_resized.save(new_dir_name)

    return im, new_dir_name

def display_img_and_result(img, rating):
    '''
    Plot the image and put the rating with it
    '''

    plt.imshow(img)
    plt.title('Rating: %s/5' % (rating))
    plt.show()

def format_outputs(prediction):

    label_encoding = {
                        '1' : [1, 0, 0, 0, 0],
                        '2' : [0, 1, 0, 0, 0],
                        '3' : [0, 0, 1, 0, 0],
                        '4' : [0, 0, 0, 1, 0],
                        '5' : [0, 0, 0, 0, 1],
    }

    # Need to turn them all into ints
    rounded_preds = []
    for i in prediction[0]:
        rounded_preds.append(int(i))

    for label in label_encoding:
        if rounded_preds == label_encoding[label]:
            true_value = label
            return true_value


def main():
    # We want to use the load model and img_to_pixels funcs from fbp

    im, new_dir_name = format_inputs('liam.jpg')

    test_img = fbp.img_to_pixels(new_dir_name) # Should be a 350*350*3 array
    test_img_reshaped = np.reshape(test_img, (1, test_img.shape[0], test_img.shape[1], test_img.shape[2]))

    # load model
    model = fbp.load_model()
    # Do we recompile?
    model.compile(loss='categorical_crossentropy', optimizer="SGD", metrics=['accuracy'])

    # Evaluate
    rating = model.predict(test_img_reshaped)
    print(rating)

    true_value = format_outputs(rating)
    print("######### RESULTS #########\n%s Rating : %s" % (new_dir_name, true_value))
    display_img_and_result(im, true_value)

if __name__ == "__main__" : main()
