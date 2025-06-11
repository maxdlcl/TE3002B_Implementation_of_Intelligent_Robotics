# Nodo ROS 2 que detecta colores de un “semáforo” (rojo, amarillo, verde)
# en la imagen recibida por /video_source/raw.  Publica:
#   • /detected_color (Bool): True si se detecta alguno de los 3 colores
#   • /color (Int8):          1-rojo, 2-amarillo, 3-verde, 0-ninguno
#
# El umbral mínimo de área de detección (min_area) es dinámico y puede
# modificarse en tiempo de ejecución con ros2 param.

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Int8
from sensor_msgs.msg import Image
from typing import Tuple            #  ←  línea nueva


import cv2
import numpy as np
from cv_bridge import CvBridge


class ColorDetector(Node):
    """Nodo que detecta rojo, amarillo y verde y publica flags."""

    def __init__(self) -> None:
        super().__init__('color_detector')

        # Parámetro dinámico ------------------------------------------
        self.declare_parameter('min_area', 1000)     # área mínima [px²]
        self.min_area = self.get_parameter('min_area').value
        # Se registra callback para cambios en min_area
        self.add_on_set_parameters_callback(self._param_cb)

        # Publicadores ------------------------------------------------
        qos = 1
        self.pub_detect = self.create_publisher(Bool, '/detected_color', qos)
        self.pub_color  = self.create_publisher(Int8, '/color',          qos)

        # Suscriptor --------------------------------------------------
        self.sub_img = self.create_subscription(Image,
                                                '/video_source/raw',
                                                self.image_callback,
                                                qos)

        self.bridge = CvBridge()
        self.get_logger().info('Color detector iniciado (topic: /video_source/raw)')

    # Callback de actualización de parámetros dinámicos
    def _param_cb(self, params):
        for p in params:
            if p.name == 'min_area' and p.type_ == p.Type.INTEGER:
                self.min_area = p.value
                self.get_logger().info(f'Nuevo min_area: {self.min_area}')
        # Se acepta siempre la actualización
        return rclpy.parameter.SetParametersResult(successful=True)

    # Callback principal de imagen
    def image_callback(self, msg: Image) -> None:
        """Convierte la imagen, detecta colores y publica resultados."""
        # 1) ROS ⇢ OpenCV
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error al convertir imagen: {e}')
            return

        # 2) BGR ⇢ HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 3) Máscaras HSV para cada color
        masks = {
            'Red':    self._mask_red(hsv),
            'Green':  cv2.inRange(hsv, (45,  50,  50), (75, 255, 255)),
            'Yellow': cv2.inRange(hsv, (20, 100, 100), (35, 255, 255)),
        }

        # 4) Procesar cada máscara
        detections = {}
        for label, mask in masks.items():
            detected, frame = self._process_mask(label, mask, frame)
            detections[label] = detected

        # 5) Publicar si se detectó cualquier color
        self.pub_detect.publish(Bool(data=any(detections.values())))

        # 6) Publicar código del color detectado (0-3)
        code = 0
        if detections['Red']:
            code = 1
        elif detections['Yellow']:
            code = 2
        elif detections['Green']:
            code = 3
        self.pub_color.publish(Int8(data=code))

        # 7) DEBUG opcional: mostrar imagen con recuadros ---------------
        # cv2.imshow("Color detector", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     rclpy.shutdown()

    # ------------------------------------------------------------------ #
    # Máscara del color rojo (dos rangos en HSV debido al hue circular)
    @staticmethod
    def _mask_red(hsv: np.ndarray) -> np.ndarray:
        lower1 = cv2.inRange(hsv, (0,   137,  10), (8,   255, 255))
        lower2 = cv2.inRange(hsv, (170, 137, 100), (180, 255, 255))
        return cv2.bitwise_or(lower1, lower2)

    # Funcion para procesar la máscara de cada color
    def _process_mask(self, label: str,
                      mask: np.ndarray,
                      frame: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Busca contornos suficientemente grandes y dibuja bounding box."""
        contours, _ = cv2.findContours(mask,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        detected = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_area:
                detected = True
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (255, 255, 255), 2)
                cv2.putText(frame, label, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2, cv2.LINE_AA)
        return detected, frame


def main() -> None:
    rclpy.init()
    node = ColorDetector()
    try:
        rclpy.spin(node)
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
