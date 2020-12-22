import numpy as np
import matplotlib.pyplot as plt

def plot_tfrecord(train_dataset, validation_dataset):
    train_image = []
    train_mask = []
    for image, mask in train_dataset.take(5):
        train_image, train_mask = image, mask
    train_mask = np.squeeze(train_mask)

    test_image = []
    test_mask = []
    for image, mask in validation_dataset.take(5):
        test_image, test_mask = image, mask
    test_mask = np.squeeze(test_mask)

    fig, ax = plt.subplots(2, 2, figsize=(20, 10))
    ax[0][0].imshow(train_image)
    ax[0][1].imshow(train_mask)
    ax[1][0].imshow(test_image)
    ax[1][1].imshow(test_mask)
    plt.show()

def plot_img_and_mask(aug_imgs, aug_msks, title='title', max_rows=2, max_cols=2, figsize=(20, 9)):
    fig, ax = plt.subplots(max_rows, max_cols, figsize=figsize)
    fig.suptitle(title, y=0.95)
    plot_count = (max_rows * max_cols) // 2
    for idx, (img, mas) in enumerate(zip(aug_imgs[:plot_count], aug_msks[:plot_count])):
        row = (idx // max_cols) * 2
        row_masks = row + 1
        col = idx % max_cols
        ax[row, col].imshow(img)
        ax[row_masks, col].imshow(mas)
    plt.show()