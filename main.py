import torch
import cv2
import pygame
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock


class FireDetector:
    def __init__(self, video_path):
        pygame.mixer.init()
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.alarm_sound = pygame.mixer.Sound('/absolute/path/to/alarm.mp3')
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.is_fire_detected = False

    def detect_fire(self, frame):
        results = self.model(frame)
        if len(results.xyxy[0]) > 0:
            return True
        else:
            return False

    def preprocess_image(self, image):
        img = cv2.resize(image, (640, 640))
        return img

    def update(self, dt):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            return

        frame = self.preprocess_image(frame)
        if self.detect_fire(frame):
            self.is_fire_detected = True
            self.alarm_sound.play()

    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.alarm_sound.stop()


class FireDetectorApp(App):
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.fire_detector = None

    def start_detection(self, instance):
        self.fire_detector = FireDetector(self.video_path)
        Clock.schedule_interval(self.fire_detector.update, 1.0 / 30.0)

    def stop_detection(self, instance):
        if self.fire_detector is not None:
            self.fire_detector.stop()
            self.fire_detector = None

    def build(self):
        layout = BoxLayout(orientation='vertical')
        image = Image(source=self.video_path)
        start_button = Button(text='Start Detection')
        start_button.bind(on_press=self.start_detection)
        stop_button = Button(text='Stop Detection')
        stop_button.bind(on_press=self.stop_detection)
        layout.add_widget(image)
        layout.add_widget(start_button)
        layout.add_widget(stop_button)
        return layout


if __name__ == "__main__":
    video_path = "demo.mp4"
    FireDetectorApp(video_path).run()
