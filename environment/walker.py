import Box2D
from Box2D.b2 import (world, polygonShape, revoluteJointDef, contactListener)
import numpy as np
import config as C

WALKER_GROUP = -1


def generate_terrain(amplitude, frequency, length=220, dx=0.5):
    """
    Gera pontos de terreno procedural usando soma de senóides.
    amplitude=0 → chão plano.
    """
    xs = np.arange(0, length, dx)
    if amplitude == 0:
        ys = np.ones_like(xs) * 1.0
    else:
        ys = 1.0 \
           + amplitude       * np.sin(2 * np.pi * frequency       * xs) \
           + amplitude * 0.5 * np.sin(2 * np.pi * frequency * 2.3 * xs + 1.1) \
           + amplitude * 0.3 * np.sin(2 * np.pi * frequency * 0.7 * xs + 2.4)
        ys = np.maximum(ys, 0.5)
    return list(zip(xs.tolist(), ys.tolist()))


class ContactDetector(contactListener):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def BeginContact(self, contact):
        fa, fb = contact.fixtureA, contact.fixtureB
        ba, bb = fa.body, fb.body
        is_ground_a = self.env._is_ground_body(ba)
        is_ground_b = self.env._is_ground_body(bb)

        # Torso toca o chão → caiu
        if (ba == self.env.torso and is_ground_b) or \
           (bb == self.env.torso and is_ground_a):
            self.env.fallen = True

        # Pé toca o chão
        for i, foot in enumerate(self.env.feet):
            if (ba == foot and is_ground_b) or \
               (bb == foot and is_ground_a):
                self.env.foot_contacts[i] = True

    def EndContact(self, contact):
        fa, fb = contact.fixtureA, contact.fixtureB
        ba, bb = fa.body, fb.body
        is_ground_a = self.env._is_ground_body(ba)
        is_ground_b = self.env._is_ground_body(bb)
        for i, foot in enumerate(self.env.feet):
            if (ba == foot and is_ground_b) or \
               (bb == foot and is_ground_a):
                self.env.foot_contacts[i] = False


def _make_fixture(body, hw, hh, density=C.DENSITY, friction=C.FRICTION):
    fd = Box2D.b2FixtureDef(
        shape=Box2D.b2PolygonShape(box=(hw, hh)),
        density=density,
        friction=friction,
        filter=Box2D.b2Filter(groupIndex=WALKER_GROUP),
    )
    return body.CreateFixture(fd)


class WalkerEnv:
    def __init__(self):
        self.world         = world(gravity=(0, -9.8), doSleep=True)
        self.ground        = None
        self.terrain_pts   = []
        self.torso         = None
        self.joints        = []   # [hip_L, knee_L, hip_R, knee_R]
        self.ankle_joints  = []   # [ankle_L, ankle_R]
        self.feet          = []
        self.foot_contacts = [False, False]
        self.fallen        = False
        self.stalled       = False
        self.prev_x        = 5.0
        self.checkpoint_x  = 5.0
        self.step_count    = 0
        self.terrain_amplitude = 0.0
        self.terrain_frequency = 0.0
        self._build_world()

    def set_curriculum(self, amplitude, frequency):
        self.terrain_amplitude = amplitude
        self.terrain_frequency = frequency

    # ──────────────────────────────────────────────────────────────────────────
    def _build_world(self):
        self._make_ground()
        self._make_walker()

    def _make_ground(self):
        """
        Cria o terreno como uma série de polígonos planos (trapézios)
        entre pontos consecutivos — muito mais estável que chainShape.
        """
        self.terrain_pts = generate_terrain(
            self.terrain_amplitude, self.terrain_frequency)

        self.ground = self.world.CreateStaticBody()

        pts = self.terrain_pts
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]

            # Centro e ângulo do segmento
            cx    = (x1 + x2) / 2
            cy    = (y1 + y2) / 2 - 0.5   # meio do trapézio (0.5 para baixo)
            angle = np.arctan2(y2 - y1, x2 - x1)
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

            body = self.world.CreateStaticBody(
                position=(cx, cy),
                angle=angle,
            )
            body.CreatePolygonFixture(
                box=(length / 2, 0.6),   # largura = comprimento do segmento, altura = 0.6
                friction=C.FRICTION,
            )
            # Guarda referência ao primeiro corpo como "ground" para detecção
            if self.ground is None or i == 0:
                self.ground = body

        # Guarda todos os corpos do terreno para detecção de contato
        self._terrain_bodies = set(self.world.bodies)

    def _ground_y_at(self, x):
        """Interpola a altura do terreno na posição x."""
        pts = self.terrain_pts
        if not pts or x <= pts[0][0]:
            return pts[0][1] if pts else 1.0
        for i in range(len(pts) - 1):
            if pts[i][0] <= x <= pts[i+1][0]:
                t = (x - pts[i][0]) / (pts[i+1][0] - pts[i][0])
                return pts[i][1] + t * (pts[i+1][1] - pts[i][1])
        return pts[-1][1]

    def _is_ground_body(self, body):
        return body in self._terrain_bodies

    def _make_walker(self):
        sx       = 5.0
        ground_y = self._ground_y_at(sx)

        foot_y  = ground_y + C.FOOT_H  / 2 + 0.05
        ankle_y = ground_y + C.FOOT_H  + 0.05
        shin_y  = ankle_y  + C.SHIN_H  / 2
        knee_y  = ankle_y  + C.SHIN_H
        thigh_y = knee_y   + C.THIGH_H / 2
        hip_y   = knee_y   + C.THIGH_H
        torso_y = hip_y    + C.TORSO_H / 2

        self.torso = self.world.CreateDynamicBody(position=(sx, torso_y))
        _make_fixture(self.torso, C.TORSO_W / 2, C.TORSO_H / 2)

        for side in (-1, 1):
            ox = side * C.LIMB_W * 0.5

            thigh = self.world.CreateDynamicBody(position=(sx + ox, thigh_y))
            _make_fixture(thigh, C.LIMB_W / 2, C.THIGH_H / 2)
            hip_joint = self.world.CreateJoint(revoluteJointDef(
                bodyA=self.torso, bodyB=thigh,
                anchor=(sx + ox, hip_y),
                lowerAngle=-1.0, upperAngle=1.0,
                enableLimit=True,
                maxMotorTorque=C.JOINT_TORQUE_LIMIT,
                motorSpeed=0.0, enableMotor=True,
            ))

            shin = self.world.CreateDynamicBody(position=(sx + ox, shin_y))
            _make_fixture(shin, C.LIMB_W / 2, C.SHIN_H / 2)
            knee_joint = self.world.CreateJoint(revoluteJointDef(
                bodyA=thigh, bodyB=shin,
                anchor=(sx + ox, knee_y),
                lowerAngle=-1.4, upperAngle=0.0,
                enableLimit=True,
                maxMotorTorque=C.JOINT_TORQUE_LIMIT,
                motorSpeed=0.0, enableMotor=True,
            ))

            foot = self.world.CreateDynamicBody(position=(sx + ox, foot_y))
            _make_fixture(foot, C.FOOT_W / 2, C.FOOT_H / 2,
                          friction=C.FRICTION * 1.5)
            ankle_joint = self.world.CreateJoint(revoluteJointDef(
                bodyA=shin, bodyB=foot,
                anchor=(sx + ox, ankle_y),
                lowerAngle=-0.4, upperAngle=0.4,
                enableLimit=True,
                maxMotorTorque=C.ANKLE_TORQUE_LIMIT,
                motorSpeed=0.0, enableMotor=True,
            ))

            self.joints.extend([hip_joint, knee_joint])
            self.ankle_joints.append(ankle_joint)
            self.feet.append(foot)

        self.world.contactListener = ContactDetector(self)

    # ──────────────────────────────────────────────────────────────────────────
    def reset(self):
        self.world.contactListener = None
        for body in list(self.world.bodies):
            self.world.DestroyBody(body)
        self.joints.clear()
        self.ankle_joints.clear()
        self.feet.clear()
        self._terrain_bodies = set()
        self.ground          = None
        self.foot_contacts   = [False, False]
        self.fallen          = False
        self.stalled         = False
        self.prev_x          = 5.0
        self.checkpoint_x    = 5.0
        self.step_count      = 0
        self._make_ground()
        self._make_walker()
        return self._get_state()

    # ──────────────────────────────────────────────────────────────────────────
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        for i, joint in enumerate(self.joints):
            joint.motorSpeed     = float(action[i]) * C.JOINT_SPEED_LIMIT
            joint.maxMotorTorque = C.JOINT_TORQUE_LIMIT

        for i, joint in enumerate(self.ankle_joints):
            joint.motorSpeed     = float(action[4 + i]) * C.JOINT_SPEED_LIMIT
            joint.maxMotorTorque = C.ANKLE_TORQUE_LIMIT

        self.world.Step(1.0 / C.FPS, 8, 3)
        self.step_count += 1

        if self.step_count % C.PROGRESS_CHECK_INTERVAL == 0:
            if self.torso.position.x - self.checkpoint_x < C.PROGRESS_MIN_DIST:
                self.stalled = True
            self.checkpoint_x = self.torso.position.x

        # Detecta queda por inclinação extrema do torso (>80°), mesmo sem contato
        if abs(self.torso.angle) > 1.4:
            self.fallen = True

        state    = self._get_state()
        reward   = self._compute_reward(action)
        done     = self.fallen or self.stalled or self.step_count >= C.MAX_STEPS_PER_EP
        distance = max(self.torso.position.x - 5.0, 0.0)

        self.prev_x = self.torso.position.x
        return state, reward, done, {
            "distance": distance,
            "stalled":  self.stalled,
            "fallen":   self.fallen,
        }

    # ──────────────────────────────────────────────────────────────────────────
    def _get_state(self):
        t  = self.torso
        j  = self.joints
        aj = self.ankle_joints

        # Lookahead: altura relativa do terreno à frente do torso
        current_y = self._ground_y_at(t.position.x)
        lookahead = [
            self._ground_y_at(t.position.x + (i + 1) * C.LOOKAHEAD_SPACING) - current_y
            for i in range(C.LOOKAHEAD_POINTS)
        ]

        return np.array([
            # Torso (2)
            t.angle,
            t.angularVelocity / 5.0,
            # Quadris e joelhos (8)
            j[0].angle / 1.5,
            j[0].speed / C.JOINT_SPEED_LIMIT,
            j[1].angle / 1.5,
            j[1].speed / C.JOINT_SPEED_LIMIT,
            j[2].angle / 1.5,
            j[2].speed / C.JOINT_SPEED_LIMIT,
            j[3].angle / 1.5,
            j[3].speed / C.JOINT_SPEED_LIMIT,
            # Contato com chão (2)
            float(self.foot_contacts[0]),
            float(self.foot_contacts[1]),
            # Velocidade linear (2)
            t.linearVelocity.x / 5.0,
            t.linearVelocity.y / 5.0,
            # Tornozelos (4)
            aj[0].angle / 0.4,
            aj[0].speed / C.JOINT_SPEED_LIMIT,
            aj[1].angle / 0.4,
            aj[1].speed / C.JOINT_SPEED_LIMIT,
            # Lookahead terreno (6)
            *lookahead,
        ], dtype=np.float32)

    def _compute_reward(self, action):
        delta_x    = self.torso.position.x - self.prev_x
        movement   = delta_x * C.REWARD_FORWARD if delta_x >= 0 \
                     else delta_x * C.PENALTY_BACKWARD
        fall       = C.PENALTY_FALL  if self.fallen  else 0.0
        stall      = C.PENALTY_STALL if self.stalled else 0.0
        torque_pen = C.PENALTY_TORQUE * float(np.sum(np.square(action)))
        hip_spread = abs(self.joints[0].angle - self.joints[2].angle)
        spread_pen = C.PENALTY_LEG_SPREAD * max(0.0, hip_spread - C.LEG_SPREAD_MAX)

        # Penalidade por inclinação do torso — desencoraja tombar de lado
        tilt      = abs(self.torso.angle)
        tilt_pen  = C.PENALTY_TILT * max(0.0, tilt - C.TORSO_TILT_MAX)

        alive   = C.REWARD_ALIVE if not (self.fallen or self.stalled) else 0.0
        vel_x   = max(0.0, float(self.torso.linearVelocity.x))
        upright = C.REWARD_UPRIGHT * max(0.0, float(np.cos(self.torso.angle))) * vel_x
        return movement + fall + stall + alive + upright - torque_pen - spread_pen - tilt_pen

    # ──────────────────────────────────────────────────────────────────────────
    @property
    def torso_x(self):
        return self.torso.position.x

    @property
    def torso_y(self):
        return self.torso.position.y