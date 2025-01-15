import Mydatasetplayersjede
from Mydatasetplayersjede import Mydataset
import torch
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import cv2

"""
def mkblkimgslist(combined_list):
    darkmaskimages = []  # BGR
    label_list = ['yourright']
    color = {'yourright': (0, 0, 255)}
    for outer in combined_list:
        for inner, label in zip(outer, label_list):
                black_img = np.zeros((720, 1280, 3), dtype=np.uint8)
                a = np.array([inner], dtype=np.int32)
                a = a.squeeze()
                a = [a.astype(np.int32)]
                print("polyfuereinbildtype", type(a))
                print("polyfuereinbildlen", len(a))
                print(f"Label: {label}, Farbe: {color[label]}")
                print(f'Laenge der Liste a: {len(a)}')
                b = cv2.fillPoly(black_img, a, color[label])
                darkmaskimages.append(b)
    print("darkmaskimageslen", len(darkmaskimages))
    return darkmaskimages



def haupt(darkmaskimages):
        # print("lenmaske:", len(darkmaskimages))
        # print("maske[idx]:", darkmaskimages[idx])

        # print("fillpoly", black_img)
        plt.imshow(darkmaskimages[0])
        plt.title("myleft?")
        plt.show()

        plt.imshow(darkmaskimages[1])
        plt.title("myright?")
        plt.show()

        plt.imshow(darkmaskimages[2])
        plt.title("yourleft?")
        plt.show()

        plt.imshow(darkmaskimages[3])
        plt.title("yourright?")
        plt.show()

        plt.imshow(darkmaskimages[168])
        plt.title("myleft?")
        plt.show()

        plt.imshow(darkmaskimages[169])
        plt.title("myright?")
        plt.show()

        plt.imshow(darkmaskimages[170])
        plt.title("yourleft?")
        plt.show()

        plt.imshow(darkmaskimages[171])
        plt.title("yourright?")
        plt.show()
"""

if __name__ == '__main__':
    path = osp.normpath(osp.join(osp.dirname(__file__), "data"))
    dataset = Mydataset(path)
    idx = 17
    # print(dataset.imgs[idx])
    print(dataset[idx])