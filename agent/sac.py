import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import config as C
from agent.sac_network import Actor, Critic
from agent.replay_buffer import ReplayBuffer
from agent.normalizer import RunningMeanStd


class SAC:
    def __init__(self):
        self.actor   = Actor()
        self.critic  = Critic()
        self.critic_target = Critic()
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.opt_actor  = optim.Adam(self.actor.parameters(),  lr=C.LR_ACTOR)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=C.LR_CRITIC)

        # Entropia automática — alpha se ajusta para manter target_entropy
        self.target_entropy = -float(C.ACTION_DIM)
        self.log_alpha = torch.tensor([-1.6], requires_grad=True)  # alpha inicial ~0.2
        self.opt_alpha = optim.Adam([self.log_alpha], lr=C.LR_ACTOR)

        self.buffer  = ReplayBuffer(C.STATE_DIM, C.ACTION_DIM,
                                    capacity=C.SAC_BUFFER_SIZE)
        self.obs_rms = RunningMeanStd(shape=(C.STATE_DIM,))

        self._step = 0

    # ──────────────────────────────────────────────────────────────────────────
    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def select_action(self, state: np.ndarray, deterministic=False):
        self.obs_rms.update(state)
        norm = self.obs_rms.normalize(state)
        s_t  = torch.FloatTensor(norm).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(s_t)
            else:
                action, _ = self.actor.sample(s_t)
        return action[0].numpy()

    def push(self, state, action, reward, next_state, done):
        norm_s  = self.obs_rms.normalize(state)
        norm_ns = self.obs_rms.normalize(next_state)
        self.buffer.push(norm_s, action, reward, norm_ns, done)

    # ──────────────────────────────────────────────────────────────────────────
    def update(self):
        if len(self.buffer) < C.SAC_BATCH_SIZE:
            return 0.0, 0.0

        self._step += 1
        states, actions, rewards, next_states, dones = \
            self.buffer.sample(C.SAC_BATCH_SIZE)

        with torch.no_grad():
            next_actions, next_log_pi = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            q_target = rewards + C.GAMMA * (1 - dones) * q_next

        # ── Critic update ──────────────────────────────────────────────────
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), C.MAX_GRAD_NORM)
        self.opt_critic.step()

        # ── Actor update ───────────────────────────────────────────────────
        new_actions, log_pi = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        actor_loss = (self.alpha * log_pi - torch.min(q1_new, q2_new)).mean()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), C.MAX_GRAD_NORM)
        self.opt_actor.step()

        # ── Alpha (entropia) update ────────────────────────────────────────
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()
        self.log_alpha.data.clamp_(-5.0, 2.0)  # alpha entre ~0.007 e ~7.4

        # ── Soft update do target critic ───────────────────────────────────
        for p, tp in zip(self.critic.parameters(),
                         self.critic_target.parameters()):
            tp.data.copy_(C.SAC_TAU * p.data + (1 - C.SAC_TAU) * tp.data)

        return critic_loss.item(), actor_loss.item()

    # ──────────────────────────────────────────────────────────────────────────
    def save(self, path="walker_sac.pt"):
        torch.save({
            "actor":        self.actor.state_dict(),
            "critic":       self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "opt_actor":    self.opt_actor.state_dict(),
            "opt_critic":   self.opt_critic.state_dict(),
            "log_alpha":    self.log_alpha.data,
            "obs_rms":      self.obs_rms.state_dict(),
        }, path)
        print(f"[SAC] Salvo → {path}")

    def load(self, path="walker_sac.pt"):
        ckpt = torch.load(path, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.opt_actor.load_state_dict(ckpt["opt_actor"])
        self.opt_critic.load_state_dict(ckpt["opt_critic"])
        self.log_alpha.data.copy_(ckpt["log_alpha"])
        if "obs_rms" in ckpt:
            self.obs_rms.load_state_dict(ckpt["obs_rms"])
        print(f"[SAC] Carregado ← {path}")
