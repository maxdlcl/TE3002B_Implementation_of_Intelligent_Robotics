#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from ultralytics import YOLO # Librería para la detección de objetos con YOLOv11
from cv_bridge import CvBridge # Conversión entre ROS y OpenCV
from sensor_msgs.msg import Image
from std_msgs.msg import Int8, Bool
from yolo_msg.msg import InferenceResult, Yolov8Inference # Mensaje personalizado para la detección de YOLOv11

# Definición del nodo de detección de señales y semáforos con YOLOv11
# Recibe imágenes de la cámara del robot, realiza la detección de objetos y publica los resultados codificados
class YoloDetection(Node):
    def __init__(self):
        super().__init__('yolo_detection_node')
        self.get_logger().info('YoloDetection Node has been started...')

        # Inicialización del modelo YOLOv11s preentrenado
        self.model = YOLO('/home/max/ros2_ws/src/final_cocos/final_cocos/coco.pt')

        self.bridge = CvBridge() # Conversión de imágenes entre ROS y OpenCV
        self.img = np.zeros((320, 180, 3), dtype=np.uint8) # Declaración de imagen vacía
        self.valid_img = False

        # Publicadores: imagen anotada, detección de color y señal
        self.yolo_img_pub = self.create_publisher(Image, '/inference_result', 10)
        self.pub_color_detected = self.create_publisher(Bool, '/detected_color', 10)
        self.pub_color = self.create_publisher(Int8, '/color', 10)
        self.pub_signal_detected = self.create_publisher(Bool, '/detected_signal', 10)
        self.pub_signal = self.create_publisher(Int8, '/signal', 10)

        # Suscriptor: imagen de la cámara del robot
        self.create_subscription(Image, '/video_source/raw', self.camera_callback, 10)

        # Timer de inferenciadame
        self.timer = self.create_timer(1/20, self.timer_callback)

        # Umbral de confianza
        self.min_confidence = 0.5
        self.detected_color = False

    # Convierte el mensaje de imagen de ROS a OpenCV y actualiza la imagen válida
    def camera_callback(self, msg):
        try:
            self.img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.valid_img = True
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            self.valid_img = False

    # Procesa la imagen, realiza la detección de objetos y publica los resultados
    def timer_callback(self):
        if not self.valid_img:
            return

        result = self.model(self.img, verbose=False, conf=self.min_confidence)
        yolo_msg = Yolov8Inference()
        yolo_msg.header.stamp = self.get_clock().now().to_msg()
        yolo_msg.header.frame_id = 'yolo_frame'

        detected_signal = False

        #
        boxes = result[0].boxes
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])

            # Construcción del mensaje de inferencia
            detection = InferenceResult()
            detection.class_name = class_name
            detection.confidence = confidence
            detection.top = int(xyxy[1])
            detection.left = int(xyxy[0])
            detection.bottom = int(xyxy[3])
            detection.right = int(xyxy[2])
            yolo_msg.yolov8_inference.append(detection)

            # Lógica de detección de semáforos por color, área y confianza
            if class_name in ["red", "yellow", "green"] and 485 <= area <= 1200:
                if confidence > 0.5:
                    color_val = {"red": 1, "yellow": 2, "green": 3}.get(class_name, 0)
                    self.pub_color_detected.publish(Bool(data=True))
                    self.pub_color.publish(Int8(data=color_val))

            else:
                self.pub_color_detected.publish(Bool(data=False))
                self.pub_color.publish(Int8(data=0))

            # Lógica de detección de señales de tráfico específicas
            if class_name in ["turn_left"]:
                if 1000 <= area <= 3000 and confidence > 0.5:
                    signal_val = {
                        "turn_left": 3,
                    }.get(class_name, 0)
                    self.pub_signal_detected.publish(Bool(data=True))
                    self.pub_signal.publish(Int8(data=signal_val))
                    detected_signal = True

                    return
                    
            elif class_name in ["go_straight", "turn_right"]:
                if 2500 <= area <= 4000 and confidence >= 0.77:
                    signal_val = {
                        "go_straight": 2,
                        "turn_right": 4
                    }.get(class_name, 0)
                    self.pub_signal_detected.publish(Bool(data=True))
                    self.pub_signal.publish(Int8(data=signal_val))
                    detected_signal = True
                    
            elif class_name in ["give_way", "roadwork_ahead"]:
                if 1500 <= area <= 7000 and 0.6 < confidence:
                    signal_val = {
                        "give_way": 5,
                        "roadwork_ahead": 6
                    }.get(class_name, 0)
                    self.pub_signal_detected.publish(Bool(data=True))
                    self.pub_signal.publish(Int8(data=signal_val))
                    detected_signal = True

            elif class_name in ["stop"]:
                if 5000 <= area and confidence > 0.7:
                    signal_val = {
                        "stop": 1
                    }.get(class_name, 0)
                    self.pub_signal_detected.publish(Bool(data=True))
                    self.pub_signal.publish(Int8(data=signal_val))
                    detected_signal = True            

        if not detected_signal:
            self.pub_signal_detected.publish(Bool(data=False))
            self.pub_signal.publish(Int8(data=0))

        # Publicación del mensaje de detección y la imagen anotada
        annotated_img = result[0].plot()
        img_msg = self.bridge.cv2_to_imgmsg(annotated_img, "bgr8")
        self.yolo_img_pub.publish(img_msg)

# Definición de la función main para iniciar el nodo
def main(args=None):
    rclpy.init(args=args)
    nodeh = YoloDetection()
    try: rclpy.spin(nodeh)
    except Exception as error: print(error)
    except KeyboardInterrupt: print("\nNode terminated by user")
    finally:
        nodeh.destroy_node()
        rclpy.try_shutdown()

# Ejecución del script principal
if __name__ == '__main__':
    main()
