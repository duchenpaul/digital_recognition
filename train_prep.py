import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
# from tqdm import tqdm
import random

import list_number_conv

DATADIR = r"C:\Users\chdu\Desktop\Portal\Other\Test yard\ML\Numbers\mnist_png"
TRAINDIR = os.path.join(DATADIR,'training')
TESTDIR = os.path.join(DATADIR,'testing')


CATEGORIES = ['0','1','2','3','4','5','6','7','8','9',]
IMG_SIZE = 28

def create_training_data():
    training_data = []
    for category in CATEGORIES:  # do dogs and cats
        path = os.path.join(TRAINDIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)
        label = list_number_conv.number2list(int(category))
        for img in os.listdir(path):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path, img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                # plt.imshow(img_array, cmap='gray')  # graph it
                # plt.show()  # display!

                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, label])
                # plt.imshow(new_array, cmap='gray')
                # plt.show()
            except Exception as e:
                print('Error Reading: ' + os.path.join(path, img))
                os.remove(os.path.join(path, img))
                pass
            # break
        # break
    random.shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def process_test_data():
    test_data = []
    path = TESTDIR  # create path to dogs and cats
    for img in os.listdir(path):  # iterate over each image per dogs and cats
        try:
            print(os.path.join(path, img))
            img_array = cv2.imread(os.path.join(path, img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
            # plt.imshow(img_array, cmap='gray')  # graph it
            # plt.show()  # display!

            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            # plt.imshow(new_array, cmap='gray')
            # plt.show()
            test_data.append([new_array, img.split('.')[0]])
        except Exception as e:
            print('Error Reading: ' + os.path.join(path, img))
            print(e)
            # os.remove(os.path.join(path, img))
            pass
        # break
    return test_data


if __name__ == '__main__':
    training_data = create_training_data()
