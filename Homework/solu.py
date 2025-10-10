import cv2
import numpy as np
import json
import os

class ColorRange:
    def __init__(self, config):
        self.ranges = config["color_ranges"]

    def get_mask(self, hsv_img, color_name):
        masks = []
        for lower, upper in self.ranges[color_name]:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv_img, lower, upper)
            masks.append(mask)
        return np.bitwise_or.reduce(masks)

class BallDetector:
    def __init__(self, config):
        self.color_range = ColorRange(config)
        self.roi = config["roi"]

    def detect_ball(self, frame):
        roi_frame = frame[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        color_names = ["red", "blue", "purple"]
        color_pixels = {}
        for color in color_names:
            mask = self.color_range.get_mask(hsv, color)
            color_pixels[color] = cv2.countNonZero(mask)
        max_color = max(color_pixels, key=color_pixels.get)
        if color_pixels[max_color] < 4450:  # 阈值可调
            return "no ball"
        return max_color + " ball"

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            status = self.detect_ball(frame)
            # 从配置中读取 roi，并在每帧上绘制矩形
            x1, y1, x2, y2 = self.roi
            # 绘制 ROI 矩形（绿色，线宽 2）
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 在左上角显示状态文本
            cv2.putText(frame, status, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.imshow("result", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

def main():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    detector = BallDetector(config)
    detector.process_video(r"res/output1.avi")

if __name__ == "__main__":
    main()