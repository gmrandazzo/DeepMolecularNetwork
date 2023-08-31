#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import argparse
import sys
import matplotlib.pyplot as plt
from keras import models
from keras.models import load_model
from keras.preprocessing import image
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=None, type=str, help='model input')
    p.add_argument('--img', default=None, type=str, help='Image Input')
    args = p.parse_args(sys.argv[1:])
    if args.model is None and args.img is None:
        print("\nUsage: %s --model [model input] --img [image file to predict] \n" % (sys.argv[0]))
        print("Example usage: ./model/visualize_layer.py --model model.h5 --img CHEMBL209060.jpg\n")
    else:
        model = load_model(args.model)
        img = image.load_img(args.img, target_size=(150, 150))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        model.summary()
        prob = model.predict(img_tensor)[-1]
        if prob < 0.5:
            print("CAT %f" % (1-prob))
        else:
            print("DOG %f" % (prob))
        layer_outputs = [layer.output for layer in model.layers[:8]]
        activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
        activations = activation_model.predict(img_tensor)

        plt.imshow(img_tensor[0])
        plt.show()

        layer_names = []
        for layer in model.layers[:8]:
            layer_names.append(layer.name)

        images_per_row = 16
        for layer_name, layer_activation in zip(layer_names, activations):
            n_features = layer_activation.shape[-1]
            size = layer_activation.shape[1]
            n_cols = n_features // images_per_row
            display_grid = np.zeros((size * n_cols, images_per_row * size))
            for col in range(n_cols):
                for row in range(images_per_row):
                    channel_image = layer_activation[0,
                                                     :, :,
                                                     col * images_per_row + row]
                    channel_image -= channel_image.mean()
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                    display_grid[col * size : (col + 1) * size,
                                 row * size : (row + 1) * size] = channel_image
            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1],
                       scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
            plt.show()

if __name__ == '__main__':
    main()
