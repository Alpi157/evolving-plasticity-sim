import numpy as np, torch, config
from creature import THETA_LEN
class LearningModule:
    def __init__(self):
        self.gamma = config.GAMMA
        self.theta_len = THETA_LEN
    def learn(self, creature, prev_obs, action, reward, new_obs):
        t_prev = torch.from_numpy(prev_obs).float()
        t_new = torch.from_numpy(new_obs).float()
        v_prev = creature.brain(t_prev)
        v_new = creature.brain(t_new)
        q_prev = v_prev[action]
        q_new = v_new.max()
        td_err = reward + self.gamma * q_new.item() - q_prev.item()
        creature.brain.zero_grad()
        q_prev.backward()
        grads = torch.cat([p.grad.view(-1) for p in creature.brain.parameters()]).numpy()
        creature.e_trace = self.gamma * creature.lmbda * creature.e_trace + grads
        delta = creature.eta * td_err * creature.e_trace if creature.hebb == 0 else creature.eta * grads
        creature.genome[:self.theta_len] += delta
        creature.theta0 += delta
        i = 0
        sz_w1 = config.HIDDEN_SIZE * config.NUM_SENSORS
        d_w1 = delta[i:i+sz_w1].reshape(config.HIDDEN_SIZE, config.NUM_SENSORS)
        i += sz_w1
        d_b1 = delta[i:i+config.HIDDEN_SIZE]
        i += config.HIDDEN_SIZE
        sz_w2 = config.NUM_OUTPUTS * config.HIDDEN_SIZE
        d_w2 = delta[i:i+sz_w2].reshape(config.NUM_OUTPUTS, config.HIDDEN_SIZE)
        i += sz_w2
        d_b2 = delta[i:i+config.NUM_OUTPUTS]
        with torch.no_grad():
            creature.brain.fc1.weight += torch.from_numpy(d_w1).float()
            creature.brain.fc1.bias += torch.from_numpy(d_b1).float()
            creature.brain.fc2.weight += torch.from_numpy(d_w2).float()
            creature.brain.fc2.bias += torch.from_numpy(d_b2).float()
