#!/usr/bin/env python3
# cmd_robot_node.py
#
# Genera /cmd_vel a partir de:
#   • /line_error                   (Float32)  ─ error lateral de la línea
#   • /follow_line      (Bool)      – velocidad normal con control lateral
#   • /reduce_velocity  (Bool)      – velocidad y kp a la mitad
#   • /stop             (Bool)      – paro total (rojo)
#   • /signal_stop      (Bool)      – paro total (señal STOP)
#   • /go_ahead         (Bool)      – avanzar recto en intersección
#   • /turn_left/right  (Bool)      – giro durante la intersección
#
# Lógica:
#   1. STOP / SIGNAL_STOP  → Twist = 0
#   2. TURN_LEFT / TURN_RIGHT → v_turn + ω_turn (±)
#   3. GO_AHEAD            → v_max recto, sin control lateral
#   4. REDUCE_VEL          → ley de control con v/2
#   5. FOLLOW_LINE         → ley de control normal
#
# Watch-dog solo se aplica cuando se necesita /line_error (FOLLOW_LINE o REDUCE_VEL).

import rclpy
from rclpy.node import Node
from enum import Enum, auto
from rclpy.clock import Clock
from rclpy.duration import Duration
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Bool, Int8

# Enum para los estados de la FSM interna
class StateFSM(Enum):
    FOLLOW_LINE  = auto()
    REDUCE_VEL   = auto()
    STOP         = auto()
    SIGNAL_STOP  = auto()
    GO_AHEAD     = auto()
    TURN_LEFT    = auto()
    TURN_RIGHT   = auto()


# Nodo que convierte el error de línea y flags en mensajes Twist para /cmd_vel.
class CmdRobotNode(Node):
    def __init__(self):
        super().__init__('cmd_robot_node')
        self.get_logger().info('Nodo cmd_robot_node iniciado')

        # Parametros del nodo
        self.declare_parameter('v_max',      0.15)   # m/s
        self.declare_parameter('v_min',      0.05)   # m/s
        self.declare_parameter('kp_ang',     0.8)    # P (FOLLOW_LINE)
        self.declare_parameter('threshold',  0.05)   # zona muerta
        self.declare_parameter('timeout',    0.5)    # s  watchdog
        self.declare_parameter('rate',      50.0)    # Hz
        self.declare_parameter('turn_speed', 0.1)   # m/s en giros
        self.declare_parameter('turn_omega', 0.8)    # rad/s giros

        # Lectura de parámetros
        self.v_max   = self.get_parameter('v_max').value
        self.v_min   = self.get_parameter('v_min').value
        self.kp_ang  = self.get_parameter('kp_ang').value
        self.thresh  = self.get_parameter('threshold').value
        self.timeout = self.get_parameter('timeout').value
        self.dt      = 1.0 / self.get_parameter('rate').value
        self.v_turn  = self.get_parameter('turn_speed').value
        self.w_turn  = self.get_parameter('turn_omega').value

        # Publicador de comandos de velocidad (/cmd_vel)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)

        # Suscripciones a mensajes de entrada
        self.create_subscription(Float32, '/line_error',   self.on_line_error,    10)
        self.create_subscription(Int8, '/line_side', self.on_line_side, 10)  # para compatibilidad

        # Flags del MainController
        self.create_subscription(Bool, '/follow_line',     self.on_follow,        10)
        self.create_subscription(Bool, '/reduce_velocity', self.on_reduce,        10)
        self.create_subscription(Bool, '/stop',            self.on_stop,          10)
        self.create_subscription(Bool, '/signal_stop',     self.on_signal_stop,   10)
        self.create_subscription(Bool, '/go_ahead',        self.on_go_ahead,      10)
        self.create_subscription(Bool, '/turn_left',       self.on_turn_left,     10)
        self.create_subscription(Bool, '/turn_right',      self.on_turn_right,    10)

        # Estado inicial y variables de control
        self.state          = StateFSM.FOLLOW_LINE
        self.error_line     = 0.0
        self.last_msg_time  = Clock().now()   # para watchdog
        self.last_side      = 0.0            # último lado detectado (+1 derecha, -1 izq)

        # Timer para el bucle de control
        self.create_timer(self.dt, self.control_loop)

    # Calñback para el error de línea detectado por última vez
    def on_line_side(self, msg: Int8):
        self.last_side = msg.data


    # Callbacks para los flags del MainController
    def on_follow(self, msg: Bool):
        if msg.data:
            self.state = StateFSM.FOLLOW_LINE
        elif self.state == StateFSM.FOLLOW_LINE:
            self.state = StateFSM.STOP        # (fallback)

    def on_reduce(self, msg: Bool):
        if msg.data:
            self.state = StateFSM.REDUCE_VEL
        elif self.state == StateFSM.REDUCE_VEL:
            self.state = StateFSM.FOLLOW_LINE

    def on_stop(self, msg: Bool):
        if msg.data:
            self.state = StateFSM.STOP
        elif self.state == StateFSM.STOP:
            self.state = StateFSM.FOLLOW_LINE

    def on_signal_stop(self, msg: Bool):
        if msg.data:
            self.state = StateFSM.SIGNAL_STOP
        elif self.state == StateFSM.SIGNAL_STOP:
            self.state = StateFSM.FOLLOW_LINE

    def on_go_ahead(self, msg: Bool):
        if msg.data:
            self.state = StateFSM.GO_AHEAD
        elif self.state == StateFSM.GO_AHEAD:
            self.state = StateFSM.FOLLOW_LINE

    def on_turn_left(self, msg: Bool):
        if msg.data:
            self.state = StateFSM.TURN_LEFT
        elif self.state == StateFSM.TURN_LEFT:
            self.state = StateFSM.FOLLOW_LINE

    def on_turn_right(self, msg: Bool):
        if msg.data:
            self.state = StateFSM.TURN_RIGHT
        elif self.state == StateFSM.TURN_RIGHT:
            self.state = StateFSM.FOLLOW_LINE

    # Callback para el error de línea
    def on_line_error(self, msg: Float32):
        self.error_line     = msg.data
        self.last_msg_time  = Clock().now()

    # Generador de mensajes Twist
    def generate_twist(self, linear: float, angular: float = 0.0) -> None:
        twist = Twist()
        twist.linear.x  = linear
        twist.angular.z = angular
        self.pub_cmd.publish(twist)

    # Ley de control proporcional con zona muerta
    def line_follow_twist(self, vmax: float, vmin: float, kp: float) -> None:
        err = abs(self.error_line)
        if err <= self.thresh:
            self.generate_twist(vmax, 0.0)
        else:
            #scale = (err - self.thresh) / (1.0 - self.thresh)
            #v = max(vmin, vmax * (1.0 - scale))
            w = -kp * self.error_line
            self.generate_twist(vmax, w)

    # Bucle de control principal
    def control_loop(self) -> None:
        k = 1.9/2.5
  

        # Selección del estado actual 
        if self.state in (StateFSM.STOP, StateFSM.SIGNAL_STOP):
            self.generate_twist(0.0, 0.0)

        elif self.state == StateFSM.TURN_LEFT:
            self.generate_twist(k * self.v_turn * 1.95,  k *self.w_turn * 1.0)
            
        elif self.state == StateFSM.TURN_RIGHT:
            self.generate_twist(k * self.v_turn * 1.95, -k *self.w_turn * 1.0)

        elif self.state == StateFSM.GO_AHEAD:
            self.generate_twist(k * self.v_max * 1.5 , self.last_side * -0.05)

        elif self.state == StateFSM.REDUCE_VEL:
            self.line_follow_twist(0.09/2, self.v_min/2, self.kp_ang/2)

        elif self.state == StateFSM.FOLLOW_LINE:
            self.line_follow_twist(0.1, self.v_min, self.kp_ang)


# Definición de la función main para iniciar el nodo
def main(args=None):
    rclpy.init(args=args)
    node = CmdRobotNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

# Ejecución del script principal
if __name__ == '__main__':
    main()
