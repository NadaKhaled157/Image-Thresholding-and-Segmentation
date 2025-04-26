import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
import cv2
import numpy as np


def compute_histogram(image):
    histogram = np.zeros(256, dtype=int)
    height, width = image.shape
    for y in range(height):
        for x in range(width):
            intensity = image[y, x]
            histogram[intensity] += 1
    return histogram


def cv2_to_pixmap(cv_img):
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb_image.shape
    bytes_per_line = 3 * width  # 3 channels (RGB)
    qimg = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # Load the .ui file
        self.original_cv_image = None
        self.grayscale_org_image = None
        self.image_path = None
        loadUi("MainWindow.ui", self)

        self.setWindowTitle("SEGMENTRON")
        self.setWindowIcon(QIcon("icon.png"))

        # Access widgets and connect signals/slots
        self.setup_connections()

    def setup_connections(self):
        # THRESHOLDING #
        self.RadioButton_thres.clicked.connect(self.activate_thresholding_mode)
        self.radioButton_optimal.clicked.connect(self.optimal_threshold)
        self.radioButton_spectral.clicked.connect(self.spectral_threshold)
        self.radioButton_otsu.clicked.connect(self.otsu_threshold)

        # Initially disable thresholding buttons
        self.radioButton_optimal.setEnabled(False)
        self.radioButton_spectral.setEnabled(False)
        self.radioButton_otsu.setEnabled(False)

        # SEGMENTATION #
        self.RadioButton_seg.clicked.connect(self.activate_segmentation_mode)

    def mouseDoubleClickEvent(self, event):
        self.load_image()

    def load_image(self):
        file_dialog = QFileDialog()
        self.image_path, _ = file_dialog.getOpenFileName(None, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if self.image_path:
            pixmap = QPixmap(self.image_path)
            if pixmap.isNull():
                print("Failed to load image for display")
                return
            self.original_cv_image = cv2.imread(self.image_path)
            self.grayscale_org_image = cv2.cvtColor(self.original_cv_image, cv2.COLOR_BGR2GRAY)
            if self.original_cv_image is None:
                print(f"Failed to load image with OpenCV: {self.image_path}")
                self.original_cv_image = None
                return
            scaled_pixmap = pixmap.scaled(
                self.Widget_Output_3.width(),
                self.Widget_Output_3.height(),
                Qt.KeepAspectRatio
            )
            self.Widget_Output_3.setAlignment(Qt.AlignCenter)
            self.Widget_Output_3.setPixmap(scaled_pixmap)

    def activate_thresholding_mode(self):
        if self.RadioButton_thres.isChecked():
            self.radioButton_optimal.setEnabled(True)
            self.radioButton_spectral.setEnabled(True)
            self.radioButton_otsu.setEnabled(True)
        else:
            self.radioButton_optimal.setEnabled(False)
            self.radioButton_spectral.setEnabled(False)
            self.radioButton_otsu.setEnabled(False)
        # Example: Handle radio button clicks
        sender = self.sender()
        print(f"Radio button clicked: {sender.text()}")

    def optimal_threshold(self, max_iterations=200):
        img = self.grayscale_org_image
        # Initialization
        top_left = img[0, 0]
        top_right = img[0, -1]
        bottom_left = img[-1, 0]
        bottom_right = img[-1, -1]

        old_threshold = np.mean([top_left, top_right, bottom_left, bottom_right])
        new_threshold = old_threshold + 1

        iteration = 0
        while iteration < max_iterations and abs(old_threshold - new_threshold) > 0.01:
            old_threshold = new_threshold
            background = img[img <= old_threshold]
            foreground = img[img > old_threshold]

            mu_bg = np.mean(background) if background.size > 0 else old_threshold - 1
            mu_fg = np.mean(foreground) if foreground.size > 0 else old_threshold + 1

            new_threshold = (mu_bg + mu_fg) / 2

            iteration += 1

        binary_image = np.where(img > new_threshold, 1, 0).astype(np.uint8)  # 0 bg & 1 fg
        pixmap = cv2_to_pixmap(binary_image * 255)
        scaled_pixmap = pixmap.scaled(
            self.Widget_Output_1.width(),
            self.Widget_Output_1.height(),
            Qt.KeepAspectRatio
        )
        self.Widget_Output_1.setAlignment(Qt.AlignCenter)
        self.Widget_Output_1.setPixmap(scaled_pixmap)

    def spectral_threshold(self):
        pass

    def otsu_threshold(self):
        histogram = compute_histogram(self.grayscale_org_image)
        probabilities = histogram/self.grayscale_org_image.size  # normalize histogram
        best_threshold = 0
        max_variance = 0

        for k in range(256):  # all possible thresholds
            bg_weight = np.sum(probabilities[:k + 1])  # from 0 to k is bg
            fg_weight = 1.0 - bg_weight
            if bg_weight == 0 or fg_weight == 0:
                continue  # skip to avoid division by 0

            # multiply each intensity by its probability
            # sum then divide weight i.e. weighted average intensity
            mu_bg = np.sum(np.arange(k + 1) * probabilities[:k + 1]) / bg_weight
            mu_fg = np.sum(np.arange(k + 1, 256) * probabilities[k + 1:]) / fg_weight
            variance = bg_weight * fg_weight * (mu_fg - mu_bg) ** 2  # sigma_b variance

            if variance > max_variance:  # greater variance i.e. better separation
                max_variance = variance
                best_threshold = k

        binary_image = (self.grayscale_org_image > best_threshold).astype(np.uint8) * 255
        # ------------------------------------------------------------ #
        # Built-in function for testing #
        # _, binary_image = cv2.threshold(
        #     self.grayscale_org_image,
        #     0,  # Threshold value (ignored for Otsu)
        #     255,  # Max value for pixels above threshold
        #     cv2.THRESH_BINARY + cv2.THRESH_OTSU  # Otsu's method
        # )
        # ------------------------------------------------------------ #
        try:
            pixmap = cv2_to_pixmap(binary_image)
            scaled_pixmap = pixmap.scaled(
                self.Widget_Output_1.width(),
                self.Widget_Output_1.height(),
                Qt.KeepAspectRatio
            )
            self.Widget_Output_1.setAlignment(Qt.AlignCenter)
            self.Widget_Output_1.setPixmap(scaled_pixmap)
        except Exception as e:
            print(e)
        return binary_image, best_threshold

    def activate_segmentation_mode(self):
        if self.RadioButton_seg.isChecked():
            self.radioButton_optimal.setEnabled(False)
            self.radioButton_spectral.setEnabled(False)
            self.radioButton_otsu.setEnabled(False)
        else:
            self.radioButton_optimal.setEnabled(True)
            self.radioButton_spectral.setEnabled(True)
            self.radioButton_otsu.setEnabled(True)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
