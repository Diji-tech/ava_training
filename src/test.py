from keras.models import load_model
import argparse
import pickle
import cv2

image = cv2.imread('./test.jpg')
output = image.copy()
image = cv2.resize(image, (256, 256))

image = image.astype("float") / 255.0

image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

# 读取模型和标签
print("-----read model and label------")
model = load_model('./output/resnet_ava_classification.model')
lb = pickle.loads(open('./output/resnet_ava_classification.pickle', "rb").read())

# 预测
preds = model.predict(image)

# 得到预测结果以及其对应的标签
i = preds.argmax(axis=1)[0]
label = lb.classes_[i]

# 在图像中把结果画出来
text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 2)
cv2.imwrite("res.jpg", output)

#  # 绘图
#  cv2.imshow("Image", output)
#  cv2.waitKey(0)
