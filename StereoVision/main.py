"""
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from time import time

from StereoVision.DeepthMap.DM import DepthMap


def set_plot(img):
    fig, axes = plt.subplots(1, 2, figsize=(10, 20))
    axes[0].imshow(img)
    axes[0].set_title('Original image')
    axes[1].imshow(img)
    axes[1].set_title('Translated Image')


if __name__ == '__main__':
    img_1_left = cv.imread('assets/im2.png', cv.COLOR_BGR2RGB)

    img_1_left_tr = (img_1_left[..., 0] * 299 / 1000) + (img_1_left[..., 0] * 299 / 1000) + (img_1_left[..., 2] * 114 / 1000)
    set_plot(img_1_left)

    img_2_right = cv.imread('assets/im6.png', cv.COLOR_BGR2RGB)

    img_2_right_tr = (img_2_right[..., 0] * 299 / 1000) + (img_2_right[..., 0] * 299 / 1000) + (img_2_right[..., 2] * 114 / 1000)
    set_plot(img_2_right)

    start = time()
    d_map = DepthMap(np.arange(50), img_1_left_tr, img_2_right_tr)
    vertices_matrix = d_map.get_vertices('L2')
    d_map.calculate_depth_map(vertices_matrix)
    end = time()

    print("Time prediction: ", end - start)

    arg_mins = d_map.f_pred.argmin(axis=1)

    depth_img = []

    for i in range(img_1_left.shape[0]):
        depth_img.append(list(d_map.s_pred[i, arg_mins[i], :]))

    depth_img = ((np.array(depth_img) / np.max(depth_img)) * 255).astype('uint8')

    plt.figure(figsize=(10, 10))
    plt.imshow(depth_img, cmap='gray')
    plt.show()
