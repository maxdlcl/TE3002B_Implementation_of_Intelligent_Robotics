#!/usr/bin/env python3
import rclpy
import cv2 as cv
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Int8, Bool

# Definición del nodo para calcular el error de línea y detectar intersecciones
# Procesa imágenes de la cámara del robot para detectar la posición de una línea respecto al centro de la imagen,
# publicando el error normalizado, el lado de la línea (izquierda o derecha) y si hay agluna intersección
class LineErrorNode(Node):

    def __init__(self) -> None:
        super().__init__('line_error_node')
        self.get_logger().info('Line error node initialized')

        # Parametros del nodo para la detección de líneas
        self.declare_parameter('thresh',              55.0)   # umbral mínimo
        self.declare_parameter('roi_height',           0.20)  # % parte inferior
        self.declare_parameter('roi_width',            0.60)  # % ancho centrado
        self.declare_parameter('rate',                30.0)   # Hz (no usado aquí)

        # Parametros para la detección de intersecciones
        self.declare_parameter('min_area',          1500.0)   # px² contorno grande
        self.declare_parameter('min_contours_int',      2)    # cuántos contornos ⇒ int

        # Lectura de parámetros
        self.thresh             = self.get_parameter('thresh').value
        self.roi_height         = self.get_parameter('roi_height').value
        self.roi_width          = self.get_parameter('roi_width').value
        self.min_area           = self.get_parameter('min_area').value
        self.min_contours_int   = self.get_parameter('min_contours_int').value

        self.last_side = None  # último lado detectado (+1 derecha, -1 izq)

        # Publicadores: error de línea, lado de la línea e intersección
        self.pub_err          = self.create_publisher(Float32, '/line_error',     10)
        self.pub_side         = self.create_publisher(Int8,    '/line_side',      10)
        self.pub_intersection = self.create_publisher(Bool,    '/intersection',   10)

        # Suscripción a la imagen de entrada
        self.bridge = CvBridge()
        self.create_subscription(Image, '/video_source/raw', self.image_callback, 10)

    # Callback para procesar la imagen recibida
    def image_callback(self, msg: Image) -> None:
        # Conversión de imagen de ROS a OpenCV
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Error al convertir imagen: {e}")
            return

        # Define la región de interés (ROI) en la parte inferior centrada
        h, w = frame.shape[:2]
        roi_h = frame[int(h * (1 - self.roi_height)) : h, :]
        roi_w = int(w * self.roi_width)
        start_x = (w - roi_w) // 2
        roi = roi_h[:, start_x : start_x + roi_w]

        # Convierte a escala de grises y aplica umbral adaptativo mediante Otsu
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        thr, _ = cv.threshold(gray, 0, 255,
                              cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

        # Binarización inversa
        _, bin_img = cv.threshold(gray, thr, 255, cv.THRESH_BINARY_INV)

        # Extrae contornos externos
        contours, _ = cv.findContours(bin_img, cv.RETR_EXTERNAL,
                                      cv.CHAIN_APPROX_SIMPLE)

        # Si no hay contornos, no hay línea
        if not contours:
            # Nada que publicar
            return

        # Verifica si hay intersección por cantidad y tamaño de contornos
        big_contours = [c for c in contours if cv.contourArea(c) > self.min_area]
        is_intersection = len(big_contours) >= self.min_contours_int
        if is_intersection:
            self.pub_intersection.publish(Bool(data=is_intersection))
            return # Se detectó intersección; no calcula error
        else:
            self.pub_intersection.publish(Bool(data=is_intersection))

        # Selecciona el contorno más grande (línea principal)
        biggest = max(contours, key=cv.contourArea)
        M = cv.moments(biggest)

        # Si el área del contorno es cero, no es válido
        if M['m00'] == 0:
            return  # contorno no válido

        # Calcula el centroide del contorno
        cx    = int(M['m10'] / M['m00'])
        w_roi = roi.shape[1]

        # Normalizar el error respecto al ancho de la ROI [-1, 1]
        error = (cx - w_roi // 2) / (w_roi // 2)

        # Determina el lado del desvío
        if error > 0:
            self.last_side = 1
        elif error < 0:
            self.last_side = -1

        # Publica error y lado
        self.pub_side.publish(Int8(data=int(self.last_side)))
        self.pub_err.publish(Float32(data=float(error)))

        # DEBUG opcional
        #cv.circle(roi, (cx, int(M['m01']/M['m00'])), 4, (0,255,0), -1)
        #cv.imshow("ROI", roi); cv.imshow("bin", bin_img); cv.waitKey(1)

    # Destructor del nodo
    def destroy_node(self) -> None:
        cv.destroyAllWindows()
        super().destroy_node()

# Definición de la función main para iniciar el nodo
def main() -> None:
    rclpy.init()
    nodeh = LineErrorNode()
    try: rclpy.spin(nodeh)
    except KeyboardInterrupt: print("\nNode terminated by user")
    finally:
        nodeh.destroy_node()
        rclpy.shutdown()

# Ejecución del script principal
if __name__ == '__main__':
    main()
