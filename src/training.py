import os
import glob
import numpy as np

import tensorflow as tf
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.models import Model

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("keras.__version__=",keras.__version__)
os.environ['CUDA_DEVICES_VISIBLE'] = '1'


def train_generator(img_path, label_path, num_limit=1000):
    images = []
    labels = []
    pics_per_category = {}

    for labels_file in glob.glob('{}/*.jpgl'.format(label_path)):
        label_name = os.path.basename(labels_file).split("_")[0]

        # check the total pic numbers with same category
        pic_nums = pics_per_category.setdefault(label_name, num_limit)
        if pic_nums <= 0:
            continue

        with open(labels_file, "r") as _fp:
            for _line in _fp:
                _line = _line.strip()
                if _line =="" or _line == None:
                    continue
                image_path = "{}/{}.jpg".format(img_path, _line)
                if os.path.exists(image_path):
                    # add the existed image with right label.
                    images.append(os.path.realpath(image_path))
                    labels.append(label_name)
                    # check the num_limit condition.
                    pics_per_category[label_name] -= 1
                    print("label_name", pics_per_category[label_name])
                    if pics_per_category[label_name] <= 0:
                        break
        pass
    return images, labels


# 主程序
def main():
    img_path = "../dataset/AVA_dataset/images/images"
    label_path = "../dataset/AVA_dataset/aesthetics_image_lists"
    (images, labels) = train_generator(img_path, label_path)

    print("load data successfully")

    # 对图像数据做scale处理
    images = np.array(images, dtype="float") / 255.0
    labels = np.array(labels)

    # 数据集切分
    (image_train, image_test, label_train, label_test) = train_test_split(images, labels, test_size=0.25, random_state=42)

    # one-hot 编码
    label_train = LabelBinarizer().fit_transform(label_train)
    label_test = LabelBinarizer().fit_transform(label_test)

    # 数据生成器处理
    data_generator = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest")
    # 输入图片为256x256，9个分类
    shape, classes = (256, 256, 3), 9
 
    # 调用keras的ResNet50模型
    model = keras.applications.resnet50.ResNet50(input_shape = shape, weights=None, classes=classes)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # 训练模型
    # training = model.fit(image_train, label_train, epochs=30, batch_size=6)
    training = model.fit_generator(data_generator(image_train, label_train, batch_size = 6),
            validation_data = (image_test, label_test), steps_per_epoch=len(image_train) // 6,
            epochs=30 )

     # 评估模型
    model.evaluate(image_test, label_test, batch_size=32)
 
    # 把训练好的模型保存到文件
    model.save('resnet_ava_classification.h5')


if __name__ == '__main__':
  main()
