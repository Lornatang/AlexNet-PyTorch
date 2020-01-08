import sys
import os
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import json
from PIL import Image

import torch
import torchvision.transforms as transforms
from alexnet import AlexNet


def classifier(image_path):
  # Open image
  img = Image.open(image_path)
  img = tfms(img).unsqueeze(0)

  # Classify with EfficientNet
  with torch.no_grad():
    logits = model(img)
  preds = torch.topk(logits, k=1).indices.squeeze(0).tolist()

  for idx in preds:
    label = labels_map[idx]
    prob = torch.softmax(logits, dim=1)[0, idx].item()
    return label, prob * 100


class picture(QWidget):
  def __init__(self):
    super(picture, self).__init__()

    self.resize(1000, 1000)
    self.setWindowTitle("Classifier tool")

    self.label = QLabel(self)
    self.label.setFixedSize(224, 224)
    self.label.move(300, 300)

    self.label.setStyleSheet("QLabel{background:white;}"
                             "QLabel{color:rgb(0,0,0);font-size:10px;font-weight:bold;font-family:宋体;}"
                             )

    btn = QPushButton(self)
    btn.setText("Open image")
    btn.move(10, 30)
    btn.clicked.connect(self.openimage)

  def openimage(self):
    imgName, imgType = QFileDialog.getOpenFileName(self, "Open image", "", "*.jpg;;*.png;;All Files(*)")
    jpg = QtGui.QPixmap(imgName).scaled(224, 224)
    self.label.setPixmap(jpg)
    text, prob = classifier(str(imgName))
    self.echo(str(text), str(prob))

  def echo(self, text, prob):
    QMessageBox.information(self, "Message", "Label :{}\nprob: {}".format(str(text), str(prob)))


if __name__ == "__main__":
  model = AlexNet().from_pretrained('alexnet')
  model.eval()
  # Preprocess image
  tfms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])

  # Load class names
  labels_map = json.load(open('labels_map.txt'))
  labels_map = [labels_map[str(i)] for i in range(1000)]

  app = QtWidgets.QApplication(sys.argv)
  my = picture()
  my.show()
  sys.exit(app.exec_())
