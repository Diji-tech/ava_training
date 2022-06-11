import os
import glob
import shutil


def train_generator(img_path, label_path, num_limit=10):
    images = []
    labels = []
    pics_per_category = {}

    for labels_file in glob.glob('{}/*.jpgl'.format(label_path)):
        if "test" in labels_file:
            print("lalal")
            continue
        name_items = os.path.basename(labels_file).split("_")
        label_name = name_items[0]


        # check the total pic numbers with same category
        pic_nums = pics_per_category.setdefault(label_name, 0)
        if pic_nums >= num_limit:
            continue

        with open(labels_file, "r") as _fp:
            for _line in _fp:
                _line = _line.strip()
                if _line =="" or _line == None:
                    continue
                image_path = "{}/{}.jpg".format(img_path, _line)
                if os.path.exists(image_path):
                    # add the existed image with right label.
                    images.append(image_path)
                    labels.append(label_name)
                    if not os.path.exists(f"tmp/{label_name}"):
                        os.makedirs(f"tmp/{label_name}")
                    shutil.copy(image_path, f"tmp/{label_name}/{_line}.jpg")
                    # check the num_limit condition.
                    pics_per_category[label_name] += 1
                    if pics_per_category[label_name] >= num_limit:
                        break
        pass
    print(">>> loading dataset = ", pics_per_category)
    return images, labels


def main():
    img_path = "../../dataset/AVA_dataset/images/images"
    label_path = "../../dataset/AVA_dataset/aesthetics_image_lists"
    (images, labels) = train_generator(img_path, label_path)

if __name__ == '__main__':
  main()
