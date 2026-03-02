import sys
import cv2
import mss
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QComboBox, QLabel
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QFont
from ultralytics import YOLO
import time

# ---------------- YOLO Worker ---------------- #
class YOLOWorker(QThread):
    results_ready = pyqtSignal(list)

    def __init__(self, model_path, imgsz=1024, conf=0.15):
        super().__init__()
        self.model = YOLO(model_path)
        self.running = False
        self.frame = None
        self.imgsz = imgsz
        self.conf = conf

    def run(self):
        while True:
            if self.running and self.frame is not None:
                results = self.model.predict(
                    source=self.frame,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    verbose=False
                )
                detections = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        label = self.model.names[cls] if cls < len(self.model.names) else str(cls)
                        detections.append((x1, y1, x2, y2, label, conf))
                self.results_ready.emit(detections)
            else:
                time.sleep(0.01)

    def start_esp(self):
        self.running = True

    def stop_esp(self):
        self.running = False

# ---------------- ESP Overlay ---------------- #
class ESPOverlay(QWidget):
    def __init__(self, model_path="best.pt", box_color=(0, 255, 0)):
        super().__init__()

        # Transparent overlay
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        screen = QApplication.primaryScreen()
        self.screen_width, self.screen_height = screen.size().width(), screen.size().height()
        self.setGeometry(0, 0, self.screen_width, self.screen_height)

        # Screen capture
        self.sct = mss.mss()
        self.monitor = {"top":0, "left":0, "width":self.screen_width, "height":self.screen_height}

        self.detections = []
        self.esp_active = False
        self.box_color = box_color  # RGB tuple

        # YOLO worker
        self.worker = YOLOWorker(model_path, imgsz=1024, conf=0.15)
        self.worker.results_ready.connect(self.update_detections)
        self.worker.start()
        print("YOLO model loaded!")

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.capture_screen)
        self.timer.start(15)  # ~66 FPS

    def toggle_esp(self):
        self.esp_active = not self.esp_active
        if self.esp_active:
            self.worker.start_esp()
        else:
            self.worker.stop_esp()
            self.detections = []

    def set_box_color(self, color_tuple):
        self.box_color = color_tuple

    def capture_screen(self):
        if self.esp_active:
            img = np.array(self.sct.grab(self.monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            self.worker.frame = frame
            self.update()

    def update_detections(self, detections):
        self.detections = detections

    def paintEvent(self, event):
        if not self.esp_active:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        bottom_middle_x = self.screen_width // 2
        bottom_y = self.screen_height

        for x1, y1, x2, y2, label, conf in self.detections:
            # Use selected color
            painter.setPen(QPen(QColor(*self.box_color), 3))
            painter.drawRect(int(x1), int(y1), int(x2-x1), int(y2-y1))

            box_bottom_center_x = int((x1+x2)/2)
            box_bottom_center_y = int(y2)
            painter.setPen(QPen(QColor(255,0,0,180),1))
            painter.drawLine(box_bottom_center_x, box_bottom_center_y, bottom_middle_x, bottom_y)

            painter.setFont(QFont('Consolas', 11, QFont.Weight.Bold))
            painter.setPen(QColor(255,255,255))
            painter.drawText(int(x1), int(y1)-5, f"{label.upper()} {int(conf*100)}%")

    def closeEvent(self, event):
        self.worker.stop_esp()
        event.accept()

# ---------------- GUI ---------------- #
class ESPGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ESP Overlay Controller")
        self.setGeometry(100,100,250,120)
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.button = QPushButton("Activate ESP")
        self.button.clicked.connect(self.toggle_esp)
        layout.addWidget(self.button)

        self.color_label = QLabel("Select Box Color:")
        layout.addWidget(self.color_label)

        self.dropdown = QComboBox()
        self.dropdown.addItem("Green", (0,255,0))
        self.dropdown.addItem("Red", (255,0,0))
        self.dropdown.addItem("Blue", (0,0,255))
        self.dropdown.addItem("Yellow", (255,255,0))
        self.dropdown.addItem("Cyan", (0,255,255))
        self.dropdown.currentIndexChanged.connect(self.change_color)
        layout.addWidget(self.dropdown)

        # ESP overlay
        self.overlay = ESPOverlay("best.pt")
        self.overlay.show()

    def toggle_esp(self):
        self.overlay.toggle_esp()
        if self.overlay.esp_active:
            self.button.setText("Deactivate ESP")
        else:
            self.button.setText("Activate ESP")

    def change_color(self):
        color_tuple = self.dropdown.currentData()
        self.overlay.set_box_color(color_tuple)

# ---------------- Main ---------------- #
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ESPGUI()
    gui.show()
    sys.exit(app.exec())
