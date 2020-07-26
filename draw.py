import matplotlib.pyplot as plt


def draw_results(test_images, test_labels, predictions, clothes, rows=3, cols=3):
    fig, axs = plt.subplots(3, 3)
    fig.suptitle('preditions_fashion mnist')
    n = 0
    for i in range(3):
        for j in range(3):
            axs[i, j].matshow(test_images[n])
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            text = clothes[sorted(enumerate(list(predictions[n])),
                                  key=lambda x: x[1], reverse=True)[0][0]]
            axs[i, j].set_title(
                f'label: {clothes[test_labels[n]]}\nprediction: {text}')
            n += 1
    plt.show()
