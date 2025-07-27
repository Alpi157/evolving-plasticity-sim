import torch,torch.nn as nn, numpy as np, config
class Brain(nn.Module):
    def __init__(self, genome: np.ndarray):
        super().__init__()
        self.fc1=nn.Linear(config.NUM_SENSORS,config.HIDDEN_SIZE)
        self.fc2=nn.Linear(config.HIDDEN_SIZE,config.NUM_OUTPUTS)
        i=0
        sz_w1=config.HIDDEN_SIZE*config.NUM_SENSORS
        W1=genome[i:i+sz_w1].reshape(config.HIDDEN_SIZE,config.NUM_SENSORS);i+=sz_w1
        b1=genome[i:i+config.HIDDEN_SIZE];i+=config.HIDDEN_SIZE
        sz_w2=config.NUM_OUTPUTS*config.HIDDEN_SIZE
        W2=genome[i:i+sz_w2].reshape(config.NUM_OUTPUTS,config.HIDDEN_SIZE);i+=sz_w2
        b2=genome[i:i+config.NUM_OUTPUTS]
        with torch.no_grad():
            self.fc1.weight.copy_(torch.from_numpy(W1).float())
            self.fc1.bias.copy_(torch.from_numpy(b1).float())
            self.fc2.weight.copy_(torch.from_numpy(W2).float())
            self.fc2.bias.copy_(torch.from_numpy(b2).float())
    def forward(self,x):
        if isinstance(x,np.ndarray):
            x=torch.from_numpy(x).float()
        h=torch.tanh(self.fc1(x))
        return torch.tanh(self.fc2(h))
