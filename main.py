import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
import cv2
import numpy as np


from PyQt5 import QtWidgets, uic

from PyQt5.QtGui import *




import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from PyQt5.QtWidgets import QLabel, QRadioButton
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import QMainWindow, QApplication, QButtonGroup, QPushButton, QSlider, QFileDialog
from PyQt5.QtWidgets import QFileDialog, QLabel
from PyQt5.QtGui import QPixmap, QImage


from Clustering import *



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
        # self.radioButton_spectral.clicked.connect(self.spectral_threshold)

    def setup_connections(self):
        # THRESHOLDING #
        self.RadioButton_thres.clicked.connect(self.activate_thresholding_mode)

        self.radioButton_5.clicked.connect(self.set_thresholding_mode)  # local
        self.radioButton_6.clicked.connect(self.set_thresholding_mode)  # global

        # self.radioButton_optimal.clicked.connect(self.optimal_threshold)
        # self.radioButton_spectral.clicked.connect(self.spectral_threshold)
        # self.radioButton_otsu.clicked.connect(self.otsu_threshold)
        self.radioButton_optimal.clicked.connect(self.apply_thresholding)
        self.radioButton_spectral.clicked.connect(self.apply_thresholding)
        self.radioButton_otsu.clicked.connect(self.apply_thresholding)

        # Initially disable thresholding buttons
        self.radioButton_5.setEnabled(False)
        self.radioButton_6.setEnabled(False)
        self.radioButton_optimal.setEnabled(False)
        self.radioButton_spectral.setEnabled(False)
        self.radioButton_otsu.setEnabled(False)

        # SEGMENTATION #
        # self.RadioButton_seg.clicked.connect(self.activate_segmentation_mode)

        # //////////////////////////////////////////////////////////////////////////////////////////////////////////


                # Create a button group and add the radio buttons
        button_group = QButtonGroup(self)
        button_group.addButton(self.radioButton_K_mean)
        button_group.addButton(self.radioButton_MeanShift)
        button_group.addButton(self.RadioButton_RegionGrowing)
        button_group.addButton(self.RadioButton_Agglo)
        
        # Connect the buttons to a function to handle their state change
        button_group.buttonClicked.connect(self.on_radio_button_clicked)



        main_group=QButtonGroup(self)
        main_group.addButton(self.RadioButton_thres)
        main_group.addButton(self.RadioButton_seg)

        main_group.buttonClicked.connect(self.on_radio_button_clicked)

        thres_btn_grp = QButtonGroup(self)
        thres_btn_grp.addButton(self.radioButton_5)
        thres_btn_grp.addButton(self.radioButton_6)

        thres_type_btn_grp = QButtonGroup(self)
        thres_type_btn_grp.addButton(self.radioButton_optimal)
        thres_type_btn_grp.addButton(self.radioButton_spectral)
        thres_type_btn_grp.addButton(self.radioButton_otsu)

        self.Slider_N_iter_Kmean.sliderReleased.connect(self.on_radio_button_clicked)
        self.Slider_N_clusters_Kmean.sliderReleased.connect(self.on_radio_button_clicked)
        self.radioButton_Gray_Img.setAutoExclusive(False)
        self.radioButton_Gray_Img.clicked.connect(self.on_radio_button_clicked)

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
        sender = self.sender()
        if sender.isChecked():
            self.radioButton_5.setEnabled(True)
            self.radioButton_6.setEnabled(True)
            # self.radioButton_optimal.setEnabled(True)
            # self.radioButton_spectral.setEnabled(True)
            # self.radioButton_otsu.setEnabled(True)
        else:
            self.radioButton_5.setEnabled(False)
            self.radioButton_6.setEnabled(False)
            # self.radioButton_optimal.setEnabled(False)
            # self.radioButton_spectral.setEnabled(False)
            # self.radioButton_otsu.setEnabled(False)

    def set_thresholding_mode(self):
        self.mode = self.sender().text().replace(" ", "")
        print(f"Thresholding Mode: {self.mode}")
        if self.RadioButton_thres.isChecked():
            self.radioButton_optimal.setEnabled(True)
            self.radioButton_spectral.setEnabled(True)
            self.radioButton_otsu.setEnabled(True)
        else:
            self.radioButton_optimal.setEnabled(False)
            self.radioButton_spectral.setEnabled(False)
            self.radioButton_otsu.setEnabled(False)
        # print(f"Radio button clicked: {sender.text()}")

    def apply_thresholding(self):
        sender = self.sender().text().replace(" ", "")
        if self.mode == "Local":
            print("Local")
            # Divide image into 4 quadrants
            height, width = self.grayscale_org_image.shape
            h_half, w_half = height // 2, width // 2
            quadrants = [
                self.grayscale_org_image[:h_half, :w_half],      # Top-left
                self.grayscale_org_image[:h_half, w_half:],      # Top-right
                self.grayscale_org_image[h_half:, :w_half],      # Bottom-left
                self.grayscale_org_image[h_half:, w_half:]       # Bottom-right
            ]
            # print(f"Quadrants: {quadrants}")
            # for i, quad in enumerate(quadrants):
            #     cv2.imwrite(f"quadrant_{i}.jpg", quad)
            # Apply thresholding to each quadrant
            binary_quadrants = []
            for i, quad in enumerate(quadrants):
                print("Inside quadrants loop")
                if sender == "Optimal":
                    print("Local Optimal")
                    binary_quad = self.optimal_threshold(quad, max_iterations=200)
                elif sender == "Spectral":
                    print("Local Spectral")
                    binary_quad = self.spectral_threshold(quad)
                elif sender == "OTSU":
                    print("Local OTSU")
                    binary_quad = self.otsu_threshold(quad)
                binary_quadrants.append(binary_quad)
                # cv2.imwrite(f"binary_quadrant_{i}.jpg", binary_quad * 255)
            # Merge quadrants
            top_row = np.hstack((binary_quadrants[0], binary_quadrants[1]))
            bottom_row = np.hstack((binary_quadrants[2], binary_quadrants[3]))
            binary_image = np.vstack((top_row, bottom_row))
            # Ensure merged image matches original dimensions (not really necessary)
            if binary_image.shape != (height, width):
                binary_image = cv2.resize(binary_image, (width, height), interpolation=cv2.INTER_NEAREST)
            # cv2.imwrite(f"local.jpg", binary_image * 255)
            self.display_results(binary_image)
        elif self.mode == "Global":
            print("Global")
            if sender == "Optimal":
                self.optimal_threshold(self.grayscale_org_image)
            elif sender == "Spectral":
                self.spectral_threshold(self.grayscale_org_image)
            elif sender == "OTSU":
                self.otsu_threshold(self.grayscale_org_image)

    def optimal_threshold(self, img, max_iterations=200):
        # img = self.grayscale_org_image
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
        self.display_results(binary_image)
        return binary_image
        # pixmap = cv2_to_pixmap(binary_image * 255)
        # scaled_pixmap = pixmap.scaled(
        #     self.Widget_Output_1.width(),
        #     self.Widget_Output_1.height(),
        #     Qt.KeepAspectRatio
        # )
        # self.Widget_Output_1.setAlignment(Qt.AlignCenter)
        # self.Widget_Output_1.setPixmap(scaled_pixmap)

    def spectral_threshold(self, img):
        print("Spectral")
        # Get the grayscale image
        # img = self.grayscale_org_image
        # Compute histogram
        histogram = compute_histogram(img)
        # Find peaks and valleys in the histogram
        peaks = []
        valleys = []
        # First and last points can't be peaks or valleys
        for i in range(1, 255):
            if histogram[i - 1] < histogram[i] and histogram[i] > histogram[i + 1]:
                peaks.append(i)
            elif histogram[i - 1] > histogram[i] and histogram[i] < histogram[i + 1]:
                valleys.append(i)

        # If we have at least one valley, use the deepest valley as threshold
        if len(valleys) > 0:
            # Find the valley between the two highest peaks
            if len(peaks) >= 2:
                # Sort peaks by height (histogram value)
                sorted_peaks = sorted(peaks, key=lambda x: histogram[x], reverse=True)
                two_highest_peaks = sorted(sorted_peaks[:2])  # Take highest two and sort by position

                # Find valleys between the two highest peaks
                between_valleys = [v for v in valleys if two_highest_peaks[0] < v < two_highest_peaks[1]]

                if between_valleys:
                    # Find the deepest valley
                    best_threshold = min(between_valleys, key=lambda x: histogram[x])
                else:
                    # If no valleys between peaks, use mean of two peaks
                    best_threshold = (two_highest_peaks[0] + two_highest_peaks[1]) // 2
            else:
                # If we have only one peak, find the lowest valley
                best_threshold = min(valleys, key=lambda x: histogram[x])
        else:
            # Fallback to mean if no clear bimodal distribution
            best_threshold = int(np.mean(img))

        # Apply threshold
        binary_image = (img > best_threshold).astype(np.uint8)

        # Display the binary image
        self.display_results(binary_image)
        # try:
        #     pixmap = cv2_to_pixmap(binary_image)
        #     scaled_pixmap = pixmap.scaled(
        #         self.Widget_Output_1.width(),
        #         self.Widget_Output_1.height(),
        #         Qt.KeepAspectRatio
        #     )
        #     self.Widget_Output_1.setAlignment(Qt.AlignCenter)
        #     self.Widget_Output_1.setPixmap(scaled_pixmap)
        # except Exception as e:
        #     print(f"Error displaying spectral threshold result: {e}")

        return binary_image

    def otsu_threshold(self, img):
        print("OTSU FUNC")
        # img = self.grayscale_org_image
        histogram = compute_histogram(img)
        probabilities = histogram / img.size  # normalize histogram
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

        binary_image = (img > best_threshold).astype(np.uint8)
        self.display_results(binary_image)
        # ------------------------------------------------------------ #
        # Built-in function for testing #
        # _, binary_image = cv2.threshold(
        #     self.grayscale_org_image,
        #     0,  # Threshold value (ignored for Otsu)
        #     255,  # Max value for pixels above threshold
        #     cv2.THRESH_BINARY + cv2.THRESH_OTSU  # Otsu's method
        # )
        # ------------------------------------------------------------ #
        # try:
        #     pixmap = cv2_to_pixmap(binary_image)
        #     scaled_pixmap = pixmap.scaled(
        #         self.Widget_Output_1.width(),
        #         self.Widget_Output_1.height(),
        #         Qt.KeepAspectRatio
        #     )
        #     self.Widget_Output_1.setAlignment(Qt.AlignCenter)
        #     self.Widget_Output_1.setPixmap(scaled_pixmap)
        # except Exception as e:
        #     print(e)
        return binary_image

    # def activate_segmentation_mode(self):
    #     if self.RadioButton_seg.isChecked():
    #         self.radioButton_optimal.setEnabled(False)
    #         self.radioButton_spectral.setEnabled(False)
    #         self.radioButton_otsu.setEnabled(False)
    #     else:
    #         self.radioButton_optimal.setEnabled(True)
    #         self.radioButton_spectral.setEnabled(True)
    #         self.radioButton_otsu.setEnabled(True)


# ////////////////////////////////////////////////////////segmentation///////////////////////////////////////////////////////

    def on_radio_button_clicked(self):
        buttons = [self.radioButton_5, self.radioButton_6,
                   self.radioButton_optimal, self.radioButton_spectral, self.radioButton_otsu]
        if self.RadioButton_seg.isChecked():
            for btn in buttons:
                btn.setEnabled(False)
                # btn.setChecked(True)
        else:
            for btn in buttons:
                btn.setEnabled(True)

            if self.radioButton_Gray_Img.isChecked():
                
                if self.radioButton_K_mean.isChecked():
                    value_iterations=self.Slider_N_iter_Kmean.value()
                    num_clusters=self.Slider_N_clusters_Kmean.value()
                    self.Label_N_iter_Kmean.setText(f"          N_Iterations : {value_iterations}   ")
                    self.Label_N_clusters_Kmean.setText(f"         Clusters :  {num_clusters}              ")


                    output_image=kmeans_image_clustering(self.grayscale_org_image , k=num_clusters  ,max_iters=value_iterations)
                    self.display_output_image(output_image)
                elif self.radioButton_MeanShift.isChecked():
                    pass
                elif self.RadioButton_RegionGrowing.isChecked():
                    pass
                elif self.RadioButton_Agglo.isChecked():
                    pass
            else:
                if self.radioButton_K_mean.isChecked():
                    
                    value_iterations=self.Slider_N_iter_Kmean.value()
                    num_clusters=self.Slider_N_clusters_Kmean.value()
                    self.Label_N_iter_Kmean.setText(f"          N_Iterations : {value_iterations}   ")
                    self.Label_N_clusters_Kmean.setText(f"         Clusters :  {num_clusters}              ")
                    output_image=kmeans_image_clustering(self.original_cv_image , k=num_clusters  ,max_iters=value_iterations)
                    output_image=cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                    self.display_output_image(output_image)
                    
                elif self.radioButton_MeanShift.isChecked():
                    pass
                elif self.RadioButton_RegionGrowing.isChecked():
                    pass
                elif self.RadioButton_Agglo.isChecked():
                    pass

    def display_output_image(self, output_img):
        height, width = output_img.shape[:2]
        if len(output_img.shape) == 2:
            bytes_per_line = width
            q_img = QImage(output_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        elif len(output_img.shape) == 3 and output_img.shape[2] == 3:
            bytes_per_line = 3 * width
            q_img = QImage(output_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:
            raise ValueError("Unsupported image format!")
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            472,
            self.Widget_Output_1.height(),
            Qt.KeepAspectRatio
        )
        self.Widget_Output_1.setPixmap(scaled_pixmap)

    def display_results(self, binary_image):
        pixmap = cv2_to_pixmap(binary_image * 255)
        scaled_pixmap = pixmap.scaled(
            self.Widget_Output_1.width(),
            self.Widget_Output_1.height(),
            Qt.KeepAspectRatio
        )
        self.Widget_Output_1.setAlignment(Qt.AlignCenter)
        self.Widget_Output_1.setPixmap(scaled_pixmap)












    # def on_gray_button_clicked_radio(self): 
    #     print(self.radioButton_Gray_Img.isChecked())      
    #     if self.radioButton_Gray_Img.isChecked():
    #         print("dllsd")
    #         self.radioButton_Gray_Img.setChecked(False)
    #         print( self.radioButton_Gray_Img.isChecked())

        
    #     self.on_radio_button_clicked()



def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
