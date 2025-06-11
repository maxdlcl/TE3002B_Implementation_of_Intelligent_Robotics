#!/usr/bin/env python3
# main_controller.py
#
# FSM que combina semáforo (rojo-amarillo-verde) y señales de tráfico:
#   • STOP              → Alto temporal (2 s)
#   • AHEAD_ONLY        → Seguir recto en la SIGUIENTE intersección (2 s)
#   • TURN_LEFT/RIGHT   → Girar en la SIGUIENTE intersección (2 s)
#   • GIVE_WAY          → Reducir velocidad mientras la señal sea visible
#   • ROAD_WORKS        → Reducir velocidad mientras la señal sea visible
#
# Flags publicados:
#   /follow_line, /reduce_velocity, /stop
#   /signal_stop, /go_ahead, /turn_left, /turn_right, /give_way, /road_works
#
# Subscripciones:
#   /detected_color (Bool)   ─ luz presente
#   /color          (Int8)   ─ 1-rojo, 2-amarillo, 3-verde
#   /detected_signal (Bool)  ─ señal presente
#   /signal          (Int8)  ─ 1-STOP, 2-AHEAD, 3-LEFT, 4-RIGHT, 5-GIVE_WAY, 6-ROAD_WORKS
#   /intersection    (Bool)  ─ True al entrar al cruce

import rclpy, time
from rclpy.node import Node
from enum import IntEnum, auto
from std_msgs.msg import Bool, Int8

# Enum para los estados de la FSM interna
class StateFSM(IntEnum):
    FOLLOW_LINE  = auto()
    REDUCE_VEL   = auto()
    STOP         = auto()
    SIGNAL_STOP  = auto()
    GO_AHEAD     = auto()
    TURN_LEFT    = auto()
    TURN_RIGHT   = auto()
    GIVE_WAY     = auto()
    ROAD_WORKS   = auto()

# Enum para las señales de tráfico
class SignalID(IntEnum):
    STOP        = 1
    AHEAD_ONLY  = 2
    TURN_LEFT   = 3
    TURN_RIGHT  = 4
    GIVE_WAY    = 5
    ROAD_WORKS  = 6


# Definición del nodo principal para el control de la FSM
class MainController(Node):
    def __init__(self):
        super().__init__('main_controller')

        # Publicadores
        self.pub_follow_line = self.create_publisher(Bool, '/follow_line',      1)
        self.pub_reduce      = self.create_publisher(Bool, '/reduce_velocity',  1)
        self.pub_stop        = self.create_publisher(Bool, '/stop',             1)
        self.pub_signal_stop = self.create_publisher(Bool, '/signal_stop',      1)

        self.pub_go_ahead    = self.create_publisher(Bool, '/go_ahead',      1)
        self.pub_turn_left   = self.create_publisher(Bool, '/turn_left',     1)
        self.pub_turn_right  = self.create_publisher(Bool, '/turn_right',    1)
        self.pub_give_way    = self.create_publisher(Bool, '/give_way',      1)
        self.pub_road_works  = self.create_publisher(Bool, '/road_works',    1)

        # Subscripciones
        self.create_subscription(Bool, '/detected_color',  self.detected_color_cb, 10)
        self.create_subscription(Int8, '/color',           self.color_cb,          10)
        self.create_subscription(Bool, '/detected_signal', self.detected_signal_cb,10)
        self.create_subscription(Int8, '/signal',          self.signal_cb,         10)
        self.create_subscription(Bool, '/intersection',    self.intersection_cb,   10)

        # FSM: flags y estado inicial
        self.detected_color   = False
        self.detected_signal  =	False
        self.intersection     = False
        self.color            = 0
        self.signal           = 0
        self.state            = StateFSM.FOLLOW_LINE

        # STOP (señal) temporal
        self.stop_duration = 2.0
        self.stop_t0       = None

        # Maniobras cronometradas
        self.pending_maneuver = 'ahead'     # 'ahead' | 'left' | 'right'
        self.in_maneuver      = False
        self.maneuver_t0      = None
        self.maneuver_duration = 2.5        # segundos

        # Flanco para /intersection
        self.last_intersection = False

        # Mensajes Bool reutilizables
        self.msg = {k: Bool() for k in
            ('f','r','s','ss','ga','tl','tr','gw','rw')}

        # Loop 10 Hz
        self.create_timer(0.1, self.loop)

    # Callbacks de subscripciones
    def detected_color_cb(self, msg):   self.detected_color  = msg.data
    def color_cb(self, msg):            self.color           = msg.data
    def detected_signal_cb(self, msg):  self.detected_signal = msg.data
    def signal_cb(self, msg):           self.signal          = msg.data
    def intersection_cb(self, msg):     self.intersection    = msg.data

    # Helper para publicar los flags básicos
    def publish_basic_flags(self):
        self.msg['f'].data = (self.state == StateFSM.FOLLOW_LINE)
        self.msg['r'].data = (self.state in
                              (StateFSM.REDUCE_VEL, StateFSM.GIVE_WAY, StateFSM.ROAD_WORKS))
        self.msg['s'].data = (self.state in (StateFSM.STOP, StateFSM.SIGNAL_STOP))
        self.pub_follow_line.publish(self.msg['f'])
        self.pub_reduce.publish(self.msg['r'])
        self.pub_stop.publish(self.msg['s'])

        self.msg['gw'].data = (self.state == StateFSM.GIVE_WAY)
        self.msg['rw'].data = (self.state == StateFSM.ROAD_WORKS)
        self.pub_give_way.publish(self.msg['gw'])
        self.pub_road_works.publish(self.msg['rw'])

    # Helper para limpiar los flags de maniobra
    def clear_turn_flags(self):
        for k, pub in (('ga',self.pub_go_ahead), ('tl',self.pub_turn_left), ('tr',self.pub_turn_right)):
            if self.msg[k].data:
                self.msg[k].data = False
                pub.publish(self.msg[k])

    # FMS loop
    def loop(self):
        now = time.time()

        # 1. Cronometraje de maniobras (GO_AHEAD, TURN_LEFT, TURN_RIGHT)
        if self.state in (StateFSM.GO_AHEAD, StateFSM.TURN_LEFT, StateFSM.TURN_RIGHT):
            if self.maneuver_t0 is not None and (now - self.maneuver_t0) >= self.maneuver_duration:
                self.clear_turn_flags()
                self.state        = StateFSM.FOLLOW_LINE
                self.in_maneuver  = False
                self.maneuver_t0  = None

                # Volver al estado de maniobra pendiente
                if self.pending_maneuver is None:
                    self.pending_maneuver = 'ahead'

        # 2. Señal STOP → Alto temporal (2 s)
        if self.state == StateFSM.SIGNAL_STOP:
            if self.stop_t0 is None:
                self.stop_t0 = now
                self.msg['ss'].data = True
                self.pub_signal_stop.publish(self.msg['ss'])
            elif now - self.stop_t0 >= self.stop_duration:
                self.stop_t0 = None
                self.msg['ss'].data = False
                self.pub_signal_stop.publish(self.msg['ss'])
                self.state = StateFSM.FOLLOW_LINE

        # 3. Semáforo rojo → STOP
        elif self.state == StateFSM.STOP:
            if self.detected_color and self.color == 3:
                self.state = StateFSM.FOLLOW_LINE

        # 4. REDUCE_VEL ───
        elif self.state == StateFSM.REDUCE_VEL:
            if self.detected_color and self.color == 1:
                self.state = StateFSM.STOP
            if (not self.detected_color) or (self.color == 3):
                self.state = StateFSM.FOLLOW_LINE

        # 5. GIVE_WAY / ROAD_WORKS
        elif self.state in (StateFSM.GIVE_WAY, StateFSM.ROAD_WORKS):
            if self.detected_color and self.color == 1:
                self.state = StateFSM.STOP
            elif not self.detected_signal:
                self.state = StateFSM.FOLLOW_LINE

        # 6. FOLLOW_LINE 
        if self.state == StateFSM.FOLLOW_LINE:
            if self.detected_color:
                if   self.color == 1: self.state = StateFSM.STOP
                elif self.color == 2: self.state = StateFSM.REDUCE_VEL

            if self.detected_signal:
                if   self.signal == SignalID.AHEAD_ONLY: self.pending_maneuver = 'ahead'
                elif self.signal == SignalID.TURN_LEFT:  self.pending_maneuver = 'left'
                elif self.signal == SignalID.TURN_RIGHT: self.pending_maneuver = 'right'
                elif self.signal == SignalID.STOP:
                    self.pending_maneuver = None
                    self.state = StateFSM.SIGNAL_STOP
                elif self.signal == SignalID.GIVE_WAY:   self.state = StateFSM.GIVE_WAY
                elif self.signal == SignalID.ROAD_WORKS: self.state = StateFSM.ROAD_WORKS

        # 7. Flanco de subida en /intersection
        rising_edge = self.intersection and not self.last_intersection
        self.last_intersection = self.intersection

        if rising_edge and self.pending_maneuver and not self.in_maneuver:
            if   self.pending_maneuver == 'ahead':
                self.state = StateFSM.GO_AHEAD
                self.msg['ga'].data = True; self.pub_go_ahead.publish(self.msg['ga'])
            elif self.pending_maneuver == 'left':
                self.state = StateFSM.TURN_LEFT
                self.msg['tl'].data = True; self.pub_turn_left.publish(self.msg['tl'])
            elif self.pending_maneuver == 'right':
                self.state = StateFSM.TURN_RIGHT
                self.msg['tr'].data = True; self.pub_turn_right.publish(self.msg['tr'])

            self.in_maneuver = True
            self.maneuver_t0 = now


        # 8. Publicación de flags básicos
        self.publish_basic_flags()


# Definición de la función main para iniciar el nodo
def main(args=None):
    rclpy.init(args=args)
    node = MainController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

# Ejecución del script principal
if __name__ == '__main__':
    main()


