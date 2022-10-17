import cv2
import numpy as np
from functools import partial
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QVBoxLayout, \
    QHBoxLayout, QApplication, QSlider, QLabel, QCheckBox, QLineEdit, QFileDialog
from pyqtgraph import ImageView
import os
import random
import json

TtoW = np.array([[1.38422491e+00, -4.33377230e-02, 5.09962994e+02],
                [-1.09480587e-03, 1.34647942e+00, 1.86430877e+02],
                [3.64006744e-06, -4.53230405e-05, 1.00000000e+00]])

H = TtoW

root = 'WZT/'

n1 = 'T'
n2 = 'W'
alpha = 0.3

im2 = cv2.imread("WZT/W2.png")
im1 = cv2.imread("WZT/T2.png")
width, height = im2.shape[1], im2.shape[0]

anns = 'app.json'
if anns:
    with open(anns, 'r') as f:
        anns = json.load(f)
anns = []

minmax_array = np.zeros((3, 3, 2), dtype=np.float32)


names = [['x_stretch', 'x_bottom', 'x_transfer'],
         ['y_right', 'y_stretch', 'y_transfer'],
         ['y_rotate_right', 'x_rotate_bottom']]

for i in range(H.shape[0]):
    for k in range(H.shape[1]):
        val = H[i, k]

        if val < 0:
            p = int(np.ceil(np.log10(abs(val))))
            mn, mx = -pow(10, (p + 1)), -pow(10, (p - 1))
            minmax_array[i, k] = (mn, mx)
        else:
            p = int(np.ceil(np.log10(val)))
            minmax_array[i, k] = (pow(10, (p - 1)), pow(10, (p + 1)))
        # cv2.createTrackbar('H{}{}_'.format(i, k)+names[i][k], window_name, 0, 1000, partial(on_change, i=i, k=k))

minmax_array[0, 0] = (0, 1e1)
minmax_array[1, 1] = (-1, 7)
minmax_array[1, 2] = (-1250, 550)
minmax_array[1, 2] = (-1250, 550)
minmax_array[0, 1] = (-2, 2)
minmax_array[1, 0] = (-2, 2)
minmax_array[0, 2] = (-950, 950)
minmax_array[2, 0] = (-1e-3, 1e-3)
minmax_array[2, 1] = (-1e-2, 1e-2)
minmax_array[:,:,0] = (-2) * np.abs(TtoW)
minmax_array[:,:,1] = (2) * np.abs(TtoW)
minmax_array_ = np.copy(minmax_array)


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].tolist()

color_dict = {}

def random_color():
    levels = range(32, 256, 32)
    return tuple(random.choice(levels) for _ in range(3))

def get_color(track_id):
    '''
    :param track_id: id of tracked objects
    :return: color if track id has unique color, otherwise, first initiliaze another color then returns that one
    '''
    if track_id in color_dict:
        return color_dict[track_id]
    else:
        color = random_color()
        iter = 0
        while color in color_dict.values() and iter < 3:
            color = random_color()
            iter += 1
        color_dict[track_id] = color
        return color


class AnotherWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.label = QLabel("Viewer Window")
        layout.addWidget(self.label)
        self.setLayout(layout)
        #######################################
        #######################################
        # Image Viewer
        #######################################
        #######################################
        self.image_view = ImageView(levelMode='rgba')
        show_histogram = False
        if show_histogram:
            self.image_view.ui.histogram.show()
            self.image_view.ui.roiBtn.show()
            self.image_view.ui.menuBtn.show()
        else:
            self.image_view.ui.histogram.hide()
            self.image_view.ui.roiBtn.hide()
            self.image_view.ui.menuBtn.hide()
        layout.addWidget(self.image_view)
        self.setMinimumSize(1000, 740)


class UnDistortionGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.params = None

        self.reset_coeff()

        self.central_widget = QWidget()
        self.layout = QVBoxLayout(self.central_widget)
        #######################################
        #######################################
        # First Row
        #######################################
        #######################################
        self.load_button = QPushButton('Load', self.central_widget)
        self.save_button = QPushButton('Save', self.central_widget)
        self.reset_button = QPushButton('Reset', self.central_widget)
        # self.load_button.clicked.connect(self.open_img)
        self.load_button.clicked.connect(self.show_image)
        self.save_button.clicked.connect(self.save)
        self.reset_button.clicked.connect(self.reset_button_method)
        hbox = QHBoxLayout()
        hbox.addWidget(self.save_button)
        hbox.addWidget(self.load_button)
        hbox.addWidget(self.reset_button)

        self.layout.addLayout(hbox)

        #######################################
        #######################################
        # Sliders for parameters
        #######################################
        #######################################
        self.sliders = []
        self.sliders_lines = []
        for ex, n_ in enumerate(names):
            self.sliders.append([])
            self.sliders_lines.append([])
            for ey, cn in enumerate(n_):
                sl = QSlider(Qt.Horizontal)
                #mn, mx = minmax_array[ex, ey]
                # sl.setRange(mn, mx)
                count = 2000
                sl.setRange(0, count)
                sl.valueChanged.connect(partial(self.change_dist_coefficients, ix=ex, iy=ey))
                hbox = QHBoxLayout()
                # self.layout.addWidget(sl)

                label = QLabel(str(ex) + str(ey) + '_' + cn)
                label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                label.setMinimumWidth(20)
                hbox.addWidget(label)
                hbox.addWidget(sl)
                # Line for minimum of range
                name_label_min = QLabel()
                name_label_min.setText('Min:')
                name_label_min.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                hbox.addWidget(name_label_min)
                line_min = QLineEdit()
                line_min.setFixedWidth(50)
                line_min.setText(str(mn))
                hbox.addWidget(line_min)
                # Line for max of range
                name_label_max = QLabel()
                name_label_max.setText('Max:')
                name_label_max.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                hbox.addWidget(name_label_max)
                line_max = QLineEdit()
                line_max.setFixedWidth(50)
                line_max.setText(str(mx))
                hbox.addWidget(line_max)

                button_for_range = QPushButton('OK', self)
                button_for_range.clicked.connect(self.set_ranges)
                hbox.addWidget(button_for_range)

                self.layout.addLayout(hbox)
                self.sliders[-1].append(sl)
                self.sliders_lines[-1].append((line_min, line_max))
        #######################################
        #######################################

        sl = QSlider(Qt.Horizontal)
        count = 2000
        sl.setRange(0, count)
        sl.valueChanged.connect(self.change_alpha)
        hbox = QHBoxLayout()
        label = QLabel("alpha")
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        label.setMinimumWidth(20)
        hbox.addWidget(label)
        hbox.addWidget(sl)

        self.layout.addLayout(hbox)
        # self.sliders.append(sl)
        #######################################
        #######################################

        self.setCentralWidget(self.central_widget)

        # self.open_img(val=0)
        self.w2 = AnotherWindow()
        self.w2.show()
        self.reset_sliders()
        self.show_image()

    def change_dist_coefficients(self, val, ix=1, iy=1):
        mn, mx = minmax_array[ix, iy]
        count = 2000
        val = val * (mx - mn) / count
        val = val + mn
        # val = int(val)
        self.params[ix, iy] = val
        self.show_image()

    def show_image(self):

        im1Reg = self.warp_with_anns(im1.copy())


        '''
        img_con1 = np.concatenate((np.concatenate((im2, im1Reg), axis=1),
                                   np.concatenate((cv2.resize(im1, (im2.shape[1], im2.shape[0])),
                                                   cv2.addWeighted(im2, alpha, im1Reg, 1-alpha, 0.)), axis=1)))
        '''
        img_con1 = np.concatenate((im1Reg, cv2.addWeighted(im2, alpha, im1Reg, 1 - alpha, 0.)), axis=1)
        for i, s in enumerate(str(self.params).split('\n')):
            cv2.putText(img_con1,
                        s,
                        (10, img_con1.shape[0] - 10 + 35 * (i - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 0),
                        4)
        """
        if self.show_original.checkState() != 0:
            c, h = self.img.shape[2], self.img.shape[0]
            image_copy = cv2.hconcat([image_copy, np.zeros((h, 55, c), dtype=image_copy.dtype), self.img])
        """
        self.w2.image_view.setImage(np.transpose(img_con1, (1, 0, 2)), autoRange=False)

    def save(self, ):
        print('saving')
        k1, k2, p1, p2, focal_length = self.params
        with open(os.path.join(self.root, self.img_name + '.txt'), 'w') as f:
            f.write(f'{k1} {k2} {p1} {p2} {focal_length}')

    def reset_coeff(self):
        self.params = np.copy(H)

    def reset_button_method(self):
        self.reset_coeff()
        self.reset_sliders()
        self.show_image()

    def reset_sliders(self):
        for ex, sl_ in enumerate(self.sliders):
            for ey, sl in enumerate(sl_):
                mn, mx = minmax_array[ex, ey]
                line_min, line_max = self.sliders_lines[ex][ey]
                count = 2000
                val = self.params[ex, ey] - mn
                val = val * count / (mx - mn)
                sl.setValue(int(val))
                line_min.setText(str(mn))
                line_max.setText(str(mx))

    def checkbox_state(self, b):
        if b.isChecked() is True:
            print(b.text() + " is selected")
        else:
            print(b.text() + " is deselected")

        if b.text() == "Hold Params":
            pass
        elif b.text() == "Show Original Image":
            self.show_image()

    def set_ranges(self):
        global minmax_array
        for ix in range(len(names)):
            for iy in range(len(names[ix])):
                line_min, line_max = self.sliders_lines[ix][iy]
                minmax_array[ix, iy] = (float(line_min.text()), float(line_max.text()))
        self.reset_sliders()

    def warp_with_anns(self, im):
        thermal_img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # [340:]
        thermal_img = thermal_img[..., None]
        thermal_img = np.concatenate((thermal_img, thermal_img, thermal_img), axis=-1)
        thermal_img = cv2.warpPerspective(thermal_img, self.params, (width, height))
        overlay_thermal = thermal_img.copy()
        ann_list = []
        for ann in anns:
            points = []
            seg = ann['segmentation'][0]
            for i in range(len(seg) // 2):
                points.extend(affine_transform(seg[i * 2:i * 2 + 2], t=self.params))
            points = ann['segmentation'][0]
            #  ann['segmentation']=[points]

            float_array = np.array([points]).reshape(-1, 2)

            # float_array[:, 0] = np.clip(float_array[:, 0], 0, 640)
            #  float_array[:, 1] = np.clip(float_array[:, 1], 0, 512)
            max_x = float_array[:, 0].max()
            min_x = float_array[:, 0].min()
            max_y = float_array[:, 1].max()
            min_y = float_array[:, 1].min()
            w = max_x - min_x
            h = max_y - min_y
            bbox = (min_x, min_y, w, h)
            area = w * h
            if w <= 0 or h <= 0:
                continue
            ann_list.append({
                'segmentation': [float_array.reshape(-1).tolist()],
                'iscrowd': ann['iscrowd'],
                'image_id': ann['image_id'],
                'category_id': ann['category_id'],
                'id': ann['id'],
                'bbox': bbox,
                'area': area,
                'track_id': ann['track_id'],
                'frame_id': ann['frame_id'],
            })

            points_int = np.array(float_array, dtype=np.int32).reshape((-1, 1, 2))
            color = get_color(ann['track_id'])

            cv2.fillPoly(overlay_thermal, [points_int], color=color)
        cv2.addWeighted(overlay_thermal, 0.3, thermal_img, 1 - 0.3, 0, thermal_img)
        return thermal_img

    def change_alpha(self, val):
        global alpha
        alpha = val / 2000
        self.show_image()


if __name__ == '__main__':
    app = QApplication([])
    window = UnDistortionGUI()
    window.setMinimumSize(640, 640)
    window.show()



    app.exit(app.exec_())
