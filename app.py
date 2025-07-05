
#Imports

import sys
from PyQt5.QtWidgets import QApplication, QWidget,QMainWindow,QPushButton,QVBoxLayout,QHBoxLayout,QLabel,QFileDialog
from PyQt5.QtGui import QIcon,QFont,QPixmap
from PyQt5.QtCore import Qt
import requests

def setup_buttons():
    select_btn = QPushButton("SELECT IMAGE")
    predict_btn = QPushButton("PREDICT")
    exit_btn = QPushButton("EXIT")

    select_btn.setStyleSheet("background-color: blue; color:white; font-size:14px;")
    predict_btn.setStyleSheet("background-color: blue; color:white; font-size:14px;")
    exit_btn.setStyleSheet("background-color: red; color:white; font-size:14px;")

    return select_btn, predict_btn,exit_btn

def setup_labels():
    image_label = QLabel("Image will appear here!")
    image_label.setAlignment(Qt.AlignCenter)
    image_label.setStyleSheet("border: 2px solid black;")
    image_label.setFixedSize(300,300)
    
    predict_label = QLabel("Predications will appear here!")
    predict_label.setAlignment(Qt.AlignCenter)
    predict_label.setStyleSheet("border: 2px solid black;")
    predict_label.setFont(QFont("Arial",10))

    return image_label, predict_label

class DemoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("GUI")
        self.setGeometry(100,100,400,400)
        self.setWindowIcon(QIcon("AI.png"))

        #BUTTONS 
        select_btn, predict_btn, exit_btn = setup_buttons() #get the buttons from function
        btn_layout = QHBoxLayout()                              #create a layout and addwidget
        btn_layout.addWidget(select_btn)
        btn_layout.addWidget(predict_btn)
        btn_layout.addWidget(exit_btn)

        select_btn.clicked.connect(self.select_image)
        exit_btn.clicked.connect(self.close)
        predict_btn.clicked.connect(self.prediction)

        #LABELS
        self.image_label , self.predict_label = setup_labels()
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        main_layout.addWidget(self.predict_label, alignment=Qt.AlignCenter)
        main_layout.addStretch()
        main_layout.addLayout(btn_layout)

        central = QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "select file", "","Images(*png *jpg *jpeg)")

        if file_path:
            pixmap = QPixmap(file_path)
            scaled = pixmap.scaled(250,250)
            self.image_label.setPixmap(scaled)
            self.current_image_path = file_path

    def prediction(self):
        url = "http://127.0.0.1:5000/predict"
        with open (self.current_image_path, "rb") as img:
            files = {"image":img}
            try:
                res = requests.post(url,files=files)
                out = res.json()
                self.predict_label.setText(f"Prediction: {out['prediction']}")
            except Exception as e:
                self.predict_label.setText("Predication not found!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DemoApp()
    window.show()
    sys.exit(app.exec_())
    



























# import sys
# import requests
# from PIL import Image
# from PyQt5.QtWidgets import (
#     QApplication, QWidget, QLabel, QPushButton,
#     QFileDialog, QVBoxLayout, QHBoxLayout
# )
# from PyQt5.QtGui import QPixmap, QFont
# from PyQt5.QtCore import Qt

# class RestApiGUI(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("REST API Image Classifier")
#         self.setGeometry(300, 100, 500, 500)

#         layout = QVBoxLayout()

#         self.image_label = QLabel("Image will appear here")
#         self.image_label.setAlignment(Qt.AlignCenter)
#         self.image_label.setFixedSize(300, 300)
#         self.image_label.setStyleSheet("border: 2px solid gray;")
#         layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

#         self.pred_label = QLabel("")
#         self.pred_label.setAlignment(Qt.AlignCenter)
#         self.pred_label.setFont(QFont("Arial", 14))
#         self.pred_label.setStyleSheet("color: green;")
#         layout.addWidget(self.pred_label)

#         btn_layout = QHBoxLayout()
#         self.select_btn = QPushButton("Select Image")
#         self.select_btn.clicked.connect(self.select_image)
#         self.predict_btn = QPushButton("Predict")
#         self.predict_btn.clicked.connect(self.send_to_api)
#         self.predict_btn.setEnabled(False)

#         btn_layout.addWidget(self.select_btn)
#         btn_layout.addWidget(self.predict_btn)
#         layout.addLayout(btn_layout)

#         self.setLayout(layout)
#         self.image_path = None

#     def select_image(self):
#         path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
#         if path:
#             self.image_path = path
#             pixmap = QPixmap(path).scaled(300, 300, Qt.KeepAspectRatio)
#             self.image_label.setPixmap(pixmap)
#             self.pred_label.setText("")
#             self.predict_btn.setEnabled(True)

#     def send_to_api(self):
#         if self.image_path:
#             url = "http://127.0.0.1:5000/predict"
#             with open(self.image_path, 'rb') as img_file:
#                 files = {'image': img_file}
#                 try:
#                     response = requests.post(url, files=files)
#                     if response.status_code == 200:
#                         result = response.json()
#                         self.pred_label.setText(f"Prediction: {result['prediction']}")
#                     else:
#                         self.pred_label.setText("Error: Unable to get prediction")
#                 except Exception as e:
#                     self.pred_label.setText(f"Request failed: {e}")

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = RestApiGUI()
#     window.show()
#     sys.exit(app.exec_())

