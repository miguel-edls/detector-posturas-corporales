from super_gradients.training import models
from super_gradients.common.object_names import Models
import cv2

yolo_nas_pose = models.get(Models.YOLO_NAS_POSE_L, pretrained_weights="coco_pose")





#yolo_nas_pose.predict_webcam()
prediction = yolo_nas_pose.predict("foto2.jpg")
prediction.show()


class CoordinateListSingleton:
    _instance = None

    def __init__(self):
        self.coordinates = []

    @classmethod
    def getInstance(cls):
        if cls._instance is None:
            cls._instance = CoordinateListSingleton()
        return cls._instance

    def add_coordinates(self, x, y, z, angle):
        self.coordinates.append({'x': x, 'y': y, 'z': z, 'angle': angle})

    def get_average(self):
        self.coordinates = self.coordinates[-5:]
        avg_x = sum([c['x'] for c in self.coordinates]) / len(self.coordinates)
        avg_y = sum([c['y'] for c in self.coordinates]) / len(self.coordinates)
        avg_z = sum([c['z'] for c in self.coordinates]) / len(self.coordinates)
        avg_angle = sum([c['angle'] for c in self.coordinates]) / len(self.coordinates)
        return avg_x, avg_y, avg_z, avg_angle

