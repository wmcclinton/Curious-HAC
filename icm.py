import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import random
import statistics

import numpy as np

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class ICMModel(nn.Module):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )

        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define Pos Embedding layer
        self.image_pos_layer = nn.Sequential(
            nn.Linear(self.embedding_size + 2, 64),
            nn.ReLU(),
            nn.Linear(64, self.embedding_size)
        )

        # Define Inverse model
        self.inverse_model = nn.Sequential(
            nn.Linear(2 * self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define Forward model
        self.forward_model = nn.Sequential(
            nn.Linear(action_space.n + self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, self.embedding_size)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def compute_embedding(self, obs):
        x = obs[0].transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        x = torch.cat([x,torch.Tensor(obs[1])],dim=1)
        x = self.image_pos_layer(x)

        return x


    def forward(self, obs, obs_, action, memory=None):
        x = self.compute_embedding(obs)

        x_ = self.compute_embedding(obs_)

        if self.use_memory:
            print("Haven't implemented yet..")
            quit()
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x
            embedding_ = x_

        if self.use_text:
            print("Haven't implemented yet..")
            quit()
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.inverse_model(torch.cat((embedding, embedding_), dim=1))
        a_dist = F.log_softmax(x, dim=1)

        # Action need to be one-hot
        x = self.forward_model(torch.cat((embedding,action), dim=1))
        s_pred = x

        return a_dist, s_pred, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]

class IntrinsicCuriosityModule():
    def __init__(self, obs_space, action_space, preprocess_obss, use_memory=False, use_text=False):
        self.memory = []
        self.eps_memory = []
        self.preprocess_obss = preprocess_obss
        self._obs = None

        self.action_space = action_space

        self.BETA = 0.2
        self.LAMBDA = 1000
        self.batch_size = 64
        self.num_epochs = 10
        self.model = ICMModel(obs_space, action_space, use_memory=False, use_text=False)

        self.num_eps = 0
        self.update_freq = 1

        self.saved_reward = []

        self.loss_inverse = nn.NLLLoss()
        self.loss_forward = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9)

    def store_transition(self, obs, obs_, action):
        self.memory.append([obs,obs_,action])

    def get_batch(self):
        mem_sample = random.sample(self.memory, self.batch_size)
        x = torch.cat([datum[0][0] for datum in mem_sample], 0)
        pos = torch.cat([torch.Tensor(datum[0][1]) for datum in mem_sample], 0)
        x_ = torch.cat([datum[1][0] for datum in mem_sample], 0)
        pos_ = torch.cat([torch.Tensor(datum[1][1]) for datum in mem_sample], 0)
        actions = torch.cat([datum[2] for datum in mem_sample],0)
        return x, pos, x_, pos_, actions

    def update_model(self):

        epoch_loss = []
        epoch_L_f = []
        epoch_L_i = []
        
        for t in range(self.num_epochs):
            # Forward pass: Compute predicted y by passing x to the model
            x, pos, x_, pos_, actions = self.get_batch()

            a_dist, s_pred, _ = self.model((x, pos), (x_, pos_), actions)

            L_f = self.loss_forward(s_pred, self.model.compute_embedding((x_, pos_)))
            
            target_a = torch.argmax(actions, dim=1).to(dtype=torch.long)
            L_i = self.loss_inverse(a_dist, target_a)

            # Compute and print loss
            loss = (1 - self.BETA) * L_i + self.BETA * L_f
            epoch_loss.append(loss.item())
            epoch_L_f.append(L_f.item())
            epoch_L_i.append(L_i.item())

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.num_eps % 100 == 0:
            print(self.num_eps, statistics.mean(epoch_loss))
            print("L_f:", round(statistics.mean(epoch_L_f),5), "L_i:", round(statistics.mean(epoch_L_i),5))
        


    def reshape_reward(self, obs, action, reward, done):
        self.eps_memory.append([obs, action, reward, done])

        r_i = 0

        # Compute Intrinsic Reward
        if self._obs is not None:
            obs_preproccessed = self.preprocess_obss([obs]).image
            _obs_preproccessed = self.preprocess_obss([self._obs]).image
            action_preproccessed = torch.zeros((1,self.action_space.n))
            action_preproccessed[0][action] = 1

            with torch.no_grad():
                _, s_pred, _ = self.model([_obs_preproccessed, self._obs["pos"]], [obs_preproccessed, obs["pos"]], action_preproccessed)
                r_i = self.loss_forward(s_pred, self.model.compute_embedding((obs_preproccessed, obs["pos"]))).item()

        self._obs = obs

        # Save both Intrinsic and Extrincic Reward every step every episode
        self.saved_reward.append([r_i, reward, done])

        if done == 1:
            #print(len(eps_memory))
            for i, _ in enumerate(self.eps_memory):
                if i > 0:
                    obs = self.preprocess_obss([self.eps_memory[i-1][0]]).image
                    pos = self.eps_memory[i-1][0]["pos"]
                    obs_ = self.preprocess_obss([self.eps_memory[i][0]]).image
                    pos_ = self.eps_memory[i][0]["pos"]
                    action = torch.zeros((1,self.action_space.n))
                    action[0][self.eps_memory[i][1]] = 1
                    self.store_transition((obs, pos), (obs_, pos_), action)

            self.eps_memory = []
            self._obs = None

            # Testing Foward and Inverse Models
            #a_dist, s_pred, _ = icm.model(icm.memory[0][0], icm.memory[0][1], icm.memory[0][2])

            if len(self.memory) > self.batch_size and self.num_eps % self.update_freq == 0:
                self.update_model()

            self.num_eps += 1

        return self.LAMBDA * reward + r_i #reward



