from ultralytics import YOLO
import cv2

class CameraObjectDetector:
    def __init__(self, model_path, camera_index=0, foc=750):
        self.model = YOLO(model_path)
        # self.cap = cv2.VideoCapture(camera_index)
        self.foc = foc
        self.real_hight_person_inch = 66.9
        self.real_hight_car_inch = 57.08
        print("=================================初始化完成======================================")
        # if not self.cap.isOpened():
        #     raise Exception("无法打开摄像头")

    def get_distance(self, real_hight_inch, h):
        '''返回真实距离'''
        dis_inch = real_hight_inch * self.foc / (h - 2)
        dis_cm = int(dis_inch * 2.54)
        return dis_cm / 100  # 返回米

    def pixel_to_camera_coords(self, x, y, z):
        """将图像坐标系下的像素坐标（x, y）和深度z（单位为m）转换为相机坐标系下的坐标（Xc, Yc, Zc）"""
        fx = 1031.985021
        fy = 1027.818956
        cx = 1020.977209
        cy = 543.713880

        xc = (x - cx) / fx
        yc = (y - cy) / fy
        zc = z
        return xc * zc, yc * zc, zc

    def draw_rectangle(self, image, x, y, x2, y2, class_label):
        cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, class_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        return image

    def show_image(self, image):
        cv2.imshow("Image", image)
        cv2.waitKey(1)

    def detect_objects(self,frame):
        detect_results = []
        results = self.model(frame)  # 对当前帧进行预测

        for r in results:
            for single_box in r.boxes:
                cx = single_box.xywh[0][0].cpu().numpy()
                cy = single_box.xywh[0][1].cpu().numpy()
                h = single_box.xywh[0][-1].cpu().numpy()

                if single_box.cls == 0:  # 检测到人
                    real_dist = self.get_distance(self.real_hight_person_inch, h)
                    coords_carm = self.pixel_to_camera_coords(cx, cy, real_dist)
                    # print("Detected: person, Distance (m):", real_dist, "Camera Coordinates:", coords_carm)
                    detect_results.append(coords_carm)
                    # frame = self.draw_rectangle(frame, int(single_box.xyxy[0][0].cpu().numpy()),
                    #                              int(single_box.xyxy[0][1].cpu().numpy()),
                    #                              int(single_box.xyxy[0][2].cpu().numpy()),
                    #                              int(single_box.xyxy[0][3].cpu().numpy()),
                    #                              str(coords_carm))

                elif single_box.cls == 2:  # 检测到车
                    real_dist = self.get_distance(self.real_hight_car_inch, h)
                    coords_carm = self.pixel_to_camera_coords(cx, cy, real_dist)
                    detect_results.append(coords_carm)
                    # print("Detected: car, Distance (m):", real_dist, "Camera Coordinates:", coords_carm)
                    # frame = self.draw_rectangle(frame, int(single_box.xyxy[0][0].cpu().numpy()),
                    #                              int(single_box.xyxy[0][1].cpu().numpy()),
                    #                              int(single_box.xyxy[0][2].cpu().numpy()),
                    #                              int(single_box.xyxy[0][3].cpu().numpy()),
                    #                              str(coords_carm))

            # self.show_image(frame)
            return detect_results



if __name__ == "__main__":
    detector = CameraObjectDetector('yolov8s.pt')
    try:
        detector.detect_objects()
    except KeyboardInterrupt:
        print("检测停止")
