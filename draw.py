import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os


def draw_results(test_images, test_labels, predictions, clothes, rows=3, cols=3):
    fig, axs = plt.subplots(3, 3)
    fig.suptitle('preditions_fashion mnist')
    n = 0
    for i in range(3):
        for j in range(3):
            axs[i, j].matshow(test_images[n])
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            text = clothes[np.argmax(predictions[n])]
            axs[i, j].set_title(
                f'label: {clothes[test_labels[n]]}\nprediction: {text}')
            n += 1
    plt.show()


def check_images(nrows=4, ncols=4, dirs=[], data=[]):
    pic_index = 0
    imgs = []

    fig = plt.gcf()
    fig.set_size_inches(ncols*4, nrows*4)

    pic_index += 8
    for i in range(len(dirs)):
        imgs += [os.path.join(dirs[i], fname)
                 for fname in data[i][pic_index-8:pic_index]]

    for i, img_path in enumerate(imgs):
        sp = plt.subplot(nrows, ncols, i+1)
        sp.axis('off')

        img = mpimg.imread(img_path)
        plt.imshow(img)
    plt.show()
