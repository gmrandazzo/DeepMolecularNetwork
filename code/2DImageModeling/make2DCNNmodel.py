#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2018-2019 gmrandazzo@gmail.com
# This file is part of DeepMolecularNetwork.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

import sys
import argparse
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras import regularizers
regularizers.l1_l2(l1=0.001, l2=0.001)


def make_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
              input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.75))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])
    return model


def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir,
                                              class_mode='sparse',
                                              **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen


def LoadDemo():
    original_dataset_dir = '/Users/marco/projects/DeepMol/examples/cats_and_dogs/train'
    base_dir = '/Users/marco/projects/DeepMol/examples/cats_and_dogs_small'
    os.mkdir(base_dir)
    train_dir = os.path.join(base_dir, 'train')
    os.mkdir(train_dir)
    validation_dir = os.path.join(base_dir, 'validation')
    os.mkdir(validation_dir)
    test_dir = os.path.join(base_dir, 'test')
    os.mkdir(test_dir)
    train_cats_dir = os.path.join(train_dir, 'cats')
    os.mkdir(train_cats_dir)
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    os.mkdir(train_dogs_dir)
    validation_cats_dir = os.path.join(validation_dir, 'cats')
    os.mkdir(validation_cats_dir)
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    os.mkdir(validation_dogs_dir)
    test_cats_dir = os.path.join(test_dir, 'cats')
    os.mkdir(test_cats_dir)
    test_dogs_dir = os.path.join(test_dir, 'dogs')
    os.mkdir(test_dogs_dir)
    # Training set cat
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)
    # Validation set cat
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)
    # Test set cat
    fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)
    # Training set dog
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)
    # Validation set dog
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)
    # Test set dog
    fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)
    return train_dir, validation_dir


def LoadDataset(fcsv):
    # Load the CSV
    d = pd.read_csv(fcsv, header=0)
    header = d.columns.tolist()

    # Prepare the dataset
    dxset = dataset()

    # Load the dx files
    for p in Path(dxpath).iterdir():
        if p.is_file() and ".dx" in str(p):
            conf_name = p.resolve().stem.split(".")[0]
            name = conf_name.split("_")[-1]
            try:
                target_val = float(d[d[header[0]] == name].values[-1][-1])
                """
                target binarization
                if IC50, EC50, Ki, Kd < 10 nM:
                    Hit compound
                else
                    Not Hit
                """
                target = 0
                if target_val < 10.:
                    target = 1
                else:
                    target = 0
                dxset.append(conf_name, str(p.absolute()), target)
            except:
                print("Error while importing %s" % (p))
                continue
        else:
            continue
    return dxset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mtype',
                   default=None, type=str,
                   help='model type: reg or class')
    p.add_argument('--csv', default=None, type=str, help='csv dataset table')
    p.add_argument('--imgpath', default=None, type=str, help='image path')
    p.add_argument('--mout', default=None, type=str, help='model output')
    args = p.parse_args(sys.argv[1:])
    if args.csv is None and args.imgpath is None and args.mtype is None and args.mout is None:
        print("\nUsage: %s --mtype [reg or class] --csv [CSV database] --imgpath [image path] --mout [keras model out]\n" % (sys.argv[0]))
    else:
        train_generator = None
        validation_generator = None
        if args.mtype == "reg":

        elif args.mtype == "class":
            train_dir, validation_dir = LoadDemo()
            train_datagen = ImageDataGenerator(rescale=1./255,
                                               rotation_range=40,
                                               width_shift_range=0.2,
                                               height_shift_range=0.2,
                                               shear_range=0.2,
                                               zoom_range=0.2,
                                               horizontal_flip=True,)

            test_datagen = ImageDataGenerator(rescale=1./255,
                                              rotation_range=40,
                                              width_shift_range=0.2,
                                              height_shift_range=0.2,
                                              shear_range=0.2,
                                              zoom_range=0.2,
                                              horizontal_flip=True,)

            train_generator = train_datagen.flow_from_directory(train_dir,
                                                                target_size=(150, 150),
                                                                batch_size=20,
                                                                class_mode='binary')
            print(train_generator.class_indices)

            validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                                    target_size=(150, 150),
                                                                    batch_size=20,
                                                                    class_mode='binary')
        else:
            print("Error! Unknown model type %s" % (args.mtype))

    model = make_model()
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=100,
                                  epochs=100,
                                  validation_data=validation_generator,
                                  validation_steps=50)

    model.save('cats_and_dogs_small_1.h5')

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


if __name__ in "__main__":
    main()
