import pygame
import pygame.gfxdraw
import numpy as np
import config as C

# ─── Paleta ───────────────────────────────────────────────────────────────────
SKY_TOP      = (20,  28,  50)
SKY_BOT      = (60,  90, 140)
GROUND_DARK  = (55, 110,  55)
GROUND_LIGHT = (75, 155,  65)
TERRAIN_FILL = (55, 110,  55)
TERRAIN_LINE = (50, 130,  50)

TORSO_COLOR  = ( 70, 140, 220)
TORSO_EDGE   = ( 40,  90, 170)
THIGH_COLOR  = ( 90, 160, 240)
SHIN_COLOR   = (120, 185, 255)
FOOT_COLOR   = (230, 175,  50)
FOOT_EDGE    = (180, 130,  30)
JOINT_COLOR  = (220, 230, 255)
JOINT_EDGE   = ( 60,  90, 150)
HEAD_COLOR   = (240, 200, 160)
HEAD_EDGE    = (180, 140, 110)
SHADOW_COLOR = (  0,   0,   0,  60)

TEXT_COLOR   = (230, 235, 255)
HUD_BG       = ( 20,  25,  40, 180)
GRAPH_RAW    = ( 60, 100,  80)
GRAPH_AVG    = ( 80, 220, 120)
STALL_COLOR  = (255, 180,  50)
FALL_COLOR   = (255,  80,  80)

PHASE_COLORS = [
    (100, 200, 100),   # fase 0 — verde (fácil)
    (200, 200,  80),   # fase 1 — amarelo
    (220, 140,  50),   # fase 2 — laranja
    (220,  80,  80),   # fase 3 — vermelho (difícil)
]


def _w2s(wx, wy, camera_x, ref_y=1.0):
    """Box2D → pixels Pygame. ref_y é o y de referência do chão."""
    ground_px = C.SCREEN_H - 80
    sx = int((wx - camera_x) * C.SCALE + C.SCREEN_W * 0.25)
    sy = int(ground_px - (wy - ref_y) * C.SCALE)
    return sx, sy


def _draw_limb(surf, body, hw, hh, fill, edge, camera_x, shadow_surf, ref_y=1.0):
    cx, cy   = body.position.x, body.position.y
    angle    = body.angle
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    corners  = []
    for lx, ly in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]:
        gx = cx + lx * cos_a - ly * sin_a
        gy = cy + lx * sin_a + ly * cos_a
        corners.append(_w2s(gx, gy, camera_x, ref_y))
    shadow = [(x + 4, y + 4) for x, y in corners]
    pygame.gfxdraw.filled_polygon(shadow_surf, shadow, SHADOW_COLOR)
    pygame.draw.polygon(surf, fill, corners)
    pygame.draw.polygon(surf, edge, corners, 2)
    return corners


def _draw_circle_joint(surf, wx, wy, r, camera_x, ref_y=1.0):
    sx, sy = _w2s(wx, wy, camera_x, ref_y)
    pygame.gfxdraw.filled_circle(surf, sx, sy, r, JOINT_COLOR)
    pygame.gfxdraw.aacircle(surf, sx, sy, r, JOINT_EDGE)


class Renderer:
    def __init__(self):
        pygame.init()
        self.screen      = pygame.display.set_mode((C.SCREEN_W, C.SCREEN_H))
        pygame.display.set_caption("Walker AI — PPO")
        self.fps_clock   = pygame.time.Clock()
        self.font        = pygame.font.SysFont("monospace", 14)
        self.font_big    = pygame.font.SysFont("monospace", 16, bold=True)
        self.shadow_surf = pygame.Surface((C.SCREEN_W, C.SCREEN_H), pygame.SRCALPHA)
        self.sky         = self._make_sky()
        self.dist_history   = []
        self.current_phase  = 0

    def _make_sky(self):
        sky = pygame.Surface((C.SCREEN_W, C.SCREEN_H))
        # Pinta toda a tela com gradiente — evita fundo preto em vales do terreno
        for y in range(C.SCREEN_H):
            t = min(y / (C.SCREEN_H - 80), 1.0)
            r = int(SKY_TOP[0] + (SKY_BOT[0] - SKY_TOP[0]) * t)
            g = int(SKY_TOP[1] + (SKY_BOT[1] - SKY_TOP[1]) * t)
            b = int(SKY_TOP[2] + (SKY_BOT[2] - SKY_TOP[2]) * t)
            pygame.draw.line(sky, (r, g, b), (0, y), (C.SCREEN_W, y))
        return sky

    # ──────────────────────────────────────────────────────────────────────────
    def render(self, env, episode, ep_reward,
               ep_distance, loss, kl=0.0, info=None):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        # Referência do terreno no ponto do torso para câmera
        ref_y    = env._ground_y_at(env.torso_x)
        camera_x = env.torso_x - C.SCREEN_W * 0.25 / C.SCALE
        ground_px = C.SCREEN_H - 80

        self.screen.blit(self.sky, (0, 0))

        # ── Terreno ────────────────────────────────────────────────────────────
        self._draw_terrain(env, camera_x, ground_px, ref_y)

        # ── Sombras ────────────────────────────────────────────────────────────
        self.shadow_surf.fill((0, 0, 0, 0))

        # ── Personagem ─────────────────────────────────────────────────────────
        self._draw_leg(env, 1, camera_x, ref_y, back=True)

        _draw_limb(self.screen, env.torso,
                   C.TORSO_W / 2, C.TORSO_H / 2,
                   TORSO_COLOR, TORSO_EDGE, camera_x, self.shadow_surf, ref_y)

        self._draw_leg(env, 0, camera_x, ref_y, back=False)

        self.screen.blit(self.shadow_surf, (0, 0))

        self._draw_head(env, camera_x, ref_y)
        self._draw_joints(env, camera_x, ref_y)

        # ── HUD + gráfico ──────────────────────────────────────────────────────
        self._draw_hud(episode, ep_reward,
                       ep_distance, loss, kl, env)
        self._draw_graph(ep_distance)

        pygame.display.flip()
        self.fps_clock.tick(C.FPS)

    # ──────────────────────────────────────────────────────────────────────────
    def _draw_terrain(self, env, camera_x, ground_px, ref_y):
        pts = env.terrain_pts
        if not pts:
            return

        # Filtra só os pontos visíveis + margem
        visible = []
        for wx, wy in pts:
            sx = int((wx - camera_x) * C.SCALE + C.SCREEN_W * 0.25)
            if -50 < sx < C.SCREEN_W + 50:
                sy = int(ground_px - (wy - ref_y) * C.SCALE)
                visible.append((sx, sy))

        if len(visible) < 2:
            return

        # Polígono preenchido até a base da tela — fecha pelos cantos inferiores
        poly = visible + [(visible[-1][0], C.SCREEN_H), (visible[0][0], C.SCREEN_H)]
        pygame.draw.polygon(self.screen, TERRAIN_FILL, poly)

        # Listras de grama no topo
        for i in range(len(visible) - 1):
            sx1, sy1 = visible[i]
            sx2, sy2 = visible[i + 1]
            mid_x = (sx1 + sx2) // 2
            stripe_w = abs(sx2 - sx1)
            if stripe_w == 0:
                continue
            if (mid_x // stripe_w) % 2 == 0:
                stripe = [visible[i], visible[i+1],
                          (sx2, sy2 + 6), (sx1, sy1 + 6)]
                pygame.draw.polygon(self.screen, GROUND_LIGHT, stripe)

        # Linha do topo do terreno
        pygame.draw.lines(self.screen, TERRAIN_LINE, False, visible, 2)

        # Marcadores de distância sobre o terreno
        for m in range(0, 500, 5):
            wx = m + 5.0
            sx = int((wx - camera_x) * C.SCALE + C.SCREEN_W * 0.25)
            if 0 < sx < C.SCREEN_W:
                wy = env._ground_y_at(wx)
                sy = int(ground_px - (wy - ref_y) * C.SCALE)
                pygame.draw.line(self.screen, (90, 160, 90),
                                 (sx, sy - 14), (sx, sy), 1)
                lbl = self.font.render(f"{m}m", True, (120, 190, 120))
                self.screen.blit(lbl, (sx - 10, sy - 28))

    # ──────────────────────────────────────────────────────────────────────────
    def _draw_leg(self, env, leg_idx, camera_x, ref_y, back=False):
        thigh_body = env.joints[leg_idx * 2].bodyB
        shin_body  = env.joints[leg_idx * 2 + 1].bodyB
        foot_b     = env.feet[leg_idx]

        dim = 40 if back else 0
        t_col = tuple(max(c - dim, 0) for c in THIGH_COLOR)
        s_col = tuple(max(c - dim, 0) for c in SHIN_COLOR)
        f_col = tuple(max(c - dim, 0) for c in FOOT_COLOR)

        _draw_limb(self.screen, thigh_body,
                   C.LIMB_W/2, C.THIGH_H/2, t_col, TORSO_EDGE,
                   camera_x, self.shadow_surf, ref_y)
        _draw_limb(self.screen, shin_body,
                   C.LIMB_W/2, C.SHIN_H/2, s_col, TORSO_EDGE,
                   camera_x, self.shadow_surf, ref_y)
        _draw_limb(self.screen, foot_b,
                   C.FOOT_W/2, C.FOOT_H/2, f_col, FOOT_EDGE,
                   camera_x, self.shadow_surf, ref_y)

    def _draw_head(self, env, camera_x, ref_y):
        tx, ty = env.torso.position.x, env.torso.position.y
        angle  = env.torso.angle
        hx = tx - np.sin(angle) * (C.TORSO_H / 2 + 0.18)
        hy = ty + np.cos(angle) * (C.TORSO_H / 2 + 0.18)
        sx, sy = _w2s(hx, hy, camera_x, ref_y)
        r = int(0.18 * C.SCALE)
        pygame.gfxdraw.filled_circle(self.screen, sx, sy, r, HEAD_COLOR)
        pygame.gfxdraw.aacircle(self.screen, sx, sy, r, HEAD_EDGE)

    def _draw_joints(self, env, camera_x, ref_y):
        for joint in env.joints:
            ax, ay = joint.anchorA
            _draw_circle_joint(self.screen, ax, ay, 4, camera_x, ref_y)
        for joint in env.ankle_joints:
            ax, ay = joint.anchorA
            _draw_circle_joint(self.screen, ax, ay, 4, camera_x, ref_y)

    # ──────────────────────────────────────────────────────────────────────────
    def _draw_hud(self, episode, ep_reward,
                  ep_distance, loss, kl, env):
        hud_surf = pygame.Surface((210, 155), pygame.SRCALPHA)
        hud_surf.fill(HUD_BG)
        self.screen.blit(hud_surf, (8, 8))

        if env.fallen:
            status_color, status_text = FALL_COLOR,  "CAIU"
        elif env.stalled:
            status_color, status_text = STALL_COLOR, "TRAVOU"
        else:
            status_color, status_text = TEXT_COLOR,  "andando"

        kl_color = FALL_COLOR  if kl > 0.02 else \
                   STALL_COLOR if kl > 0.01 else TEXT_COLOR

        # Fase do curriculum
        phase_idx = next(
            (i for i, p in reversed(list(enumerate(C.CURRICULUM)))
             if episode >= p["episode"]), 0)
        phase_names = ["Plano", "Leve", "Médio", "Difícil"]
        phase_col   = PHASE_COLORS[min(phase_idx, len(PHASE_COLORS)-1)]
        phase_name  = phase_names[min(phase_idx, len(phase_names)-1)]

        lines = [
            (f"EP {episode}  |  step {env.step_count}/{C.MAX_STEPS_PER_EP}", TEXT_COLOR),
            (f"Dist   {ep_distance:.2f} m",              TEXT_COLOR),
            (f"Reward {ep_reward:.1f}",                   TEXT_COLOR),
            (f"Loss   {loss:.4f}  KL {kl:.4f}",          kl_color),
            (f"Status {status_text}",                     status_color),
            (f"Fase   {phase_idx} — {phase_name}",        phase_col),
        ]
        for i, (txt, col) in enumerate(lines):
            surf = self.font.render(txt, True, col)
            self.screen.blit(surf, (14, 14 + i * 22))

    # ──────────────────────────────────────────────────────────────────────────
    def _draw_graph(self, current_dist):
        self.dist_history.append(current_dist)
        if len(self.dist_history) > 200:
            self.dist_history.pop(0)

        gw, gh = 260, 120
        gx, gy = C.SCREEN_W - gw - 10, 10

        g_surf = pygame.Surface((gw, gh), pygame.SRCALPHA)
        g_surf.fill((20, 25, 40, 180))
        self.screen.blit(g_surf, (gx, gy))
        pygame.draw.rect(self.screen, (80, 90, 120), (gx, gy, gw, gh), 1)

        lbl = self.font_big.render("distância / episódio", True, TEXT_COLOR)
        self.screen.blit(lbl, (gx + 6, gy + 4))

        # Linhas verticais para cada fase do curriculum
        for p in C.CURRICULUM[1:]:
            if len(self.dist_history) > 0:
                total_eps = max(p["episode"], 1)
                frac = min(p["episode"] / max(
                    p["episode"] + 200, 1), 1.0)
                px_phase = gx + int(frac * gw)
                if gx < px_phase < gx + gw:
                    pygame.draw.line(self.screen, (100, 100, 140),
                                     (px_phase, gy + 20), (px_phase, gy + gh - 4), 1)

        if len(self.dist_history) < 2:
            return

        max_d = max(self.dist_history) or 1.0

        def to_px(i, d):
            px = gx + int(i / max(len(self.dist_history) - 1, 1) * (gw - 10)) + 5
            py = gy + gh - 6 - int(d / max_d * (gh - 26))
            return px, py

        raw = [to_px(i, d) for i, d in enumerate(self.dist_history)]
        if len(raw) >= 2:
            pygame.draw.lines(self.screen, GRAPH_RAW, False, raw, 1)

        window = 20
        if len(self.dist_history) >= window:
            avg_pts = [
                to_px(i, np.mean(self.dist_history[i - window + 1: i + 1]))
                for i in range(window - 1, len(self.dist_history))
            ]
            if len(avg_pts) >= 2:
                pygame.draw.lines(self.screen, GRAPH_AVG, False, avg_pts, 2)

        max_lbl = self.font.render(f"máx {max_d:.1f}m", True, GRAPH_AVG)
        self.screen.blit(max_lbl, (gx + 6, gy + gh - 18))

    def close(self):
        pygame.quit()