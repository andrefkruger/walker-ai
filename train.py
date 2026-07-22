import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
train.py

Uso:
    python train.py               # treina do zero com visualização
    python train.py --no-render   # treina sem janela (mais rápido)
    python train.py --load        # continua treinando do melhor checkpoint
    python train.py --watch       # só assiste, sem treinar
"""

import argparse
import collections

import config as C
from environment.walker import WalkerEnv
from agent.sac import SAC


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--load",      action="store_true")
    p.add_argument("--watch",     action="store_true")
    p.add_argument("--episodes",  type=int, default=10000)
    return p.parse_args()


def get_terrain_params(phase_idx):
    phase = C.CURRICULUM[phase_idx]
    return phase["amplitude"], phase["frequency"]


def main():
    args  = parse_args()
    env   = WalkerEnv()
    agent = SAC()

    # ── Modo watch ─────────────────────────────────────────────────────────────
    if args.watch:
        agent.load("walker_sac_best.pt")
        from render.renderer import Renderer
        renderer = Renderer()
        print("Modo assistir — Ctrl+C para sair")
        episode = 0
        try:
            while True:
                episode += 1
                amp, freq = get_terrain_params(len(C.CURRICULUM) - 1)
                env.set_curriculum(amp, freq)
                state = env.reset()
                done  = False
                ep_distance = 0.0
                while not done:
                    action = agent.select_action(state, deterministic=True)
                    state, _, done, info = env.step(action)
                    ep_distance = info["distance"]
                    renderer.render(env, episode, 0.0, ep_distance, 0.0, 0.0, info)
                print(f"Ep {episode} | dist={ep_distance:.2f}m")
        except (KeyboardInterrupt, SystemExit):
            renderer.close()
        return

    # ── Modo treino ────────────────────────────────────────────────────────────
    if args.load:
        agent.load("walker_sac_best.pt")

    renderer = None
    if not args.no_render:
        from render.renderer import Renderer
        renderer = Renderer()

    total_steps  = 0
    best_dist    = 0.0
    last_loss    = 0.0
    current_phase = 0
    recent_dists  = collections.deque(maxlen=C.CURRICULUM_WINDOW)

    print("=" * 65)
    print("  Walker AI — SAC + Curriculum por Capacidade")
    print("=" * 65)
    for i, p in enumerate(C.CURRICULUM):
        print(f"  Fase {i}: amp={p['amplitude']:.2f}  freq={p['frequency']:.3f}"
              f"  → avança com best≥{p['advance_dist']:.0f}m")
    print(f"  Warmup: {C.SAC_WARMUP_STEPS} steps aleatórios")
    print("=" * 65)

    try:
        for episode in range(1, args.episodes + 1):

            # ── Curriculum ────────────────────────────────────────────────────
            if (current_phase < len(C.CURRICULUM) - 1
                    and len(recent_dists) == C.CURRICULUM_WINDOW
                    and max(recent_dists) >= C.CURRICULUM[current_phase]["advance_dist"]):
                current_phase += 1
                amp, freq = get_terrain_params(current_phase)
                print(f"\n  ▶ Fase {current_phase} ativada "
                      f"(best={max(recent_dists):.1f}m → "
                      f"amp={amp:.2f}, freq={freq:.3f})\n")
                recent_dists.clear()

            amp, freq = get_terrain_params(current_phase)
            env.set_curriculum(amp, freq)

            state       = env.reset()
            ep_reward   = 0.0
            ep_distance = 0.0
            done        = False

            while not done:
                # Warmup: ações aleatórias para preencher o buffer
                if total_steps < C.SAC_WARMUP_STEPS:
                    import numpy as np
                    action = np.random.uniform(-1, 1, C.ACTION_DIM).astype(np.float32)
                else:
                    action = agent.select_action(state)

                next_state, reward, done, info = env.step(action)
                agent.push(state, action, reward, next_state, done)

                # Update a cada step (off-policy)
                if total_steps >= C.SAC_WARMUP_STEPS:
                    c_loss, _ = agent.update()
                    last_loss = c_loss

                state        = next_state
                ep_reward   += reward
                ep_distance  = info["distance"]
                total_steps += 1

                if renderer:
                    renderer.render(env, episode, ep_reward, ep_distance,
                                    last_loss, agent.alpha, info)

            end = "caiu"   if info.get("fallen")  else \
                  "travou" if info.get("stalled") else "tempo"

            recent_dists.append(ep_distance)

            if episode % 50 == 0:
                best = max(recent_dists) if recent_dists else 0.0
                print(f"Ep {episode:5d} | fase={current_phase} | "
                      f"dist={ep_distance:6.2f}m | best={best:5.2f}m | "
                      f"reward={ep_reward:7.1f} | "
                      f"α={agent.alpha:.4f} | steps={total_steps} | fim={end}")

            if ep_distance > best_dist:
                best_dist = ep_distance
                agent.save("walker_sac_best.pt")
                print(f"  ★ Novo recorde: {best_dist:.2f}m (ep {episode})")

            if episode % C.SAVE_INTERVAL == 0:
                agent.save(f"walker_sac_ep{episode}.pt")

    except KeyboardInterrupt:
        print("\nInterrompido.")
    finally:
        agent.save("walker_sac.pt")
        if renderer:
            renderer.close()
        print(f"\nMelhor distância: {best_dist:.2f} m")


if __name__ == "__main__":
    main()
