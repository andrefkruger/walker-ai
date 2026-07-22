# ─── Configurações da Simulação ───────────────────────────────────────────────
FPS         = 60
SCALE       = 30.0
SCREEN_W    = 1200
SCREEN_H    = 600

# ─── Personagem ───────────────────────────────────────────────────────────────
TORSO_W     = 0.6
TORSO_H     = 0.8
LIMB_W      = 0.25
THIGH_H     = 0.6
SHIN_H      = 0.55
FOOT_W      = 0.4
FOOT_H      = 0.15
DENSITY     = 1.0
FRICTION    = 0.8

JOINT_TORQUE_LIMIT  = 150.0
ANKLE_TORQUE_LIMIT  = 60.0   # tornozelos mais fracos que quadril/joelho
JOINT_SPEED_LIMIT   = 6.0

# ─── Espaço de Estado / Ação ──────────────────────────────────────────────────
# 14 base + 4 tornozelos (ângulo+vel × 2) + 6 lookahead = 24
LOOKAHEAD_POINTS  = 6      # pontos de terreno à frente
LOOKAHEAD_SPACING = 0.5    # metros entre cada ponto
STATE_DIM   = 24
ACTION_DIM  = 6            # quadril×2 + joelho×2 + tornozelo×2

# ─── Recompensa ───────────────────────────────────────────────────────────────
REWARD_FORWARD      =  2.0
PENALTY_BACKWARD    =  3.0
PENALTY_FALL        = -5.0
PENALTY_STALL       =  -5.0
PENALTY_TORQUE      =  0.0005 # reduzido: agora temos 6 juntas em vez de 4
LEG_SPREAD_MAX      =  1.4
PENALTY_LEG_SPREAD  =  0.2

# Penalidade por inclinação do torso
# Se o torso inclinar mais que TILT_MAX radianos (~40°), penaliza
# Isso evita que ele aprenda a "cair de lado" como estratégia
TORSO_TILT_MAX      =  1.0   # ~57 graus — tolerante durante aprendizado
PENALTY_TILT        =  0.1   # pequeno por step — evita acumular penalidade em episódios longos
REWARD_ALIVE        =  0.05  # bônus por step vivo — incentiva sobrevivência
REWARD_UPRIGHT      =  0.20

# ─── Timeout por progresso ────────────────────────────────────────────────────
PROGRESS_CHECK_INTERVAL = 250
PROGRESS_MIN_DIST       = 0.2
MAX_STEPS_PER_EP        = 3000

# ─── Curriculum Learning ──────────────────────────────────────────────────────
# Cada fase define a amplitude e frequência do terreno.
# amplitude = altura máxima das colinas em metros
# frequency = quão próximas ficam as colinas (menor = mais suave)
# episode_threshold = a partir de qual episódio esta fase começa
#
# Fase 0: chão plano — aprende a andar
# Fase 1: ondulações leves — aprende a se recuperar
# Fase 2: terreno médio — generaliza
# Fase 3: terreno desafiador — maximiza distância

CURRICULUM = [
    {"episode": 0,    "amplitude": 0.00, "frequency": 0.00,  "advance_dist": 2.0},  # plano
    {"episode": 500,  "amplitude": 0.05, "frequency": 0.03,  "advance_dist": 2.0},  # micro-ondas
    {"episode": 1000, "amplitude": 0.10, "frequency": 0.04,  "advance_dist": 2.5},  # suave
    {"episode": 1500, "amplitude": 0.20, "frequency": 0.05,  "advance_dist": 3.0},  # médio
    {"episode": 2500, "amplitude": 0.35, "frequency": 0.065, "advance_dist": 3.0},  # difícil
    {"episode": 4000, "amplitude": 0.50, "frequency": 0.08,  "advance_dist": 9999}, # máximo
]
CURRICULUM_WINDOW = 50

# ─── Rede Neural (maior) ──────────────────────────────────────────────────────
HIDDEN_DIM   = 256
HIDDEN_LAYERS = 2
LR_ACTOR     = 3e-4
LR_CRITIC    = 1e-3

# ─── PPO ──────────────────────────────────────────────────────────────────────
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.2
ENTROPY_COEF    = 0.05
VALUE_COEF      = 0.5
MAX_GRAD_NORM   = 0.5
PPO_EPOCHS      = 5
BATCH_SIZE      = 64
ROLLOUT_STEPS   = 2048

SAVE_INTERVAL   = 200

# ─── SAC ──────────────────────────────────────────────────────────────────────
SAC_BUFFER_SIZE    = 500_000   # transições armazenadas
SAC_BATCH_SIZE     = 256       # amostras por update
SAC_UPDATES_PER_STEP = 1       # updates por step do ambiente
SAC_WARMUP_STEPS   = 5_000     # steps aleatórios antes de treinar
SAC_TAU            = 0.005     # soft update do target critic