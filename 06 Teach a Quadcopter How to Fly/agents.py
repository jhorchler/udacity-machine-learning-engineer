import numpy as np
import torch
import torch.optim as optimizers
import torch.nn.functional as F
from typing import Dict
from tasks import Task
from models import Actor, Critic
from utils import ReplayBuffer

# run on GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PolicySearchAgent:
    def __init__(self, task):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

        self.w = np.random.normal(
            size=(self.state_size, self.action_size),
            # weights for simple linear policy: state_space x action_space
            scale=(self.action_range / (
                    2 * self.state_size)))  # start producing actions
        # in a decent range

        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = 0.1

        # Episode variables
        self.total_reward = 0.0
        self.count = 0
        self.score = 0.0
        self.reset_episode()

    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        return state

    def step(self, reward, done):
        # Save experience / reward
        self.total_reward += reward
        self.count += 1

        # Learn, if at end of episode
        if done:
            self.learn()

    def act(self, state):
        # Choose action based on given state and policy
        action = np.dot(state, self.w)  # simple linear policy
        return action

    def learn(self):
        # Learn by random policy search, using a reward-based score
        self.score = self.total_reward / float(
            self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_w = self.w
            self.noise_scale = max(0.5 * self.noise_scale, 0.01)
        else:
            self.w = self.best_w
            self.noise_scale = min(2.0 * self.noise_scale, 3.2)
        self.w = self.w + self.noise_scale * np.random.normal(
            size=self.w.shape)  # equal noise in all directions


class TD3Agent:
    """Implements Twin Delayed Deep Deterministic Policy Gradients (TD3)
       Paper:
        https://arxiv.org/abs/1802.09477
       OpenAI Spinningup:
        https://spinningup.openai.com/en/latest/algorithms/td3.html
    """

    def __init__(self, task: Task, parameters: Dict) -> None:
        """Creates a new TD3Agent

        Args:
            task (Task): The task to learn.
            parameters (Dict): the dictionary holding hyper parameters
        """
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.config = parameters
        self.lr_actor = self.config.get('LR_ACTOR', 10e-3)
        self.lr_critic = self.config.get('LR_CRITIC', self.lr_actor)
        self.weight_decay = self.config.get('WEIGHT_DECAY', 0)

        # PSEUDO CODE:
        #   initialize critic networks Q1 and Q2 with random parameters
        self.critic_1 = Critic(state_size=self.state_size,
                               action_size=self.action_size,
                               conf=self.config).to(device)
        self.critic_2 = Critic(state_size=self.state_size,
                               action_size=self.action_size,
                               conf=self.config).to(device)

        # PSEUDO CODE:
        #   initialize actor network PI with random parameters
        self.actor = Actor(self.state_size, self.action_size,
                           self.action_high, conf=self.config).to(device)

        # PSEUDO CODE:
        #   initialize target networks and set parameters from local nets
        self.critic_1_target = Critic(state_size=self.state_size,
                                      action_size=self.action_size,
                                      conf=self.config).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target = Critic(state_size=self.state_size,
                                      action_size=self.action_size,
                                      conf=self.config).to(device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.actor_target = Actor(self.state_size, self.action_size,
                                  self.action_high, conf=self.config).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # NON PSEUDO CODE:
        #   initialize optimizers
        #   Paper section 6.1 states that Adam was used with learning rate
        #   of 10e-3, optional weight decay is set to 0 per default: that's the
        #   default in torch too
        self.critic_1_optimizer = optimizers.Adam(self.critic_1.parameters(),
                                                  lr=self.lr_critic,
                                                  weight_decay=self.weight_decay)
        self.critic_2_optimizer = optimizers.Adam(self.critic_2.parameters(),
                                                  lr=self.lr_critic,
                                                  weight_decay=self.weight_decay)
        self.actor_optimizer = optimizers.Adam(self.actor.parameters(),
                                               lr=self.lr_actor,
                                               weight_decay=self.weight_decay)

    def reset(self) -> np.ndarray:
        """resets the task"""
        state = self.task.reset()
        return state

    def act(self, state: np.ndarray) -> np.ndarray:
        """Returns an action based on the policy used by Actor."""
        state = np.reshape(state, [-1, self.state_size])
        state = torch.tensor(state, dtype=torch.float).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, memory: ReplayBuffer, episode_steps: int) -> None:
        """Updates the models.

        Args:
            memory (ReplayBuffer): Buffer to sample experiences from
            episode_steps (int): number of steps done in the episode
        """

        batch_size = self.config.get('BATCH_SIZE', 100)
        discount = self.config.get('DISCOUNT', 0.99)
        tau = self.config.get('TAU', 0.05)
        policy_noise = self.config.get('POLICY_NOISE', 0.2)
        noise_clip = self.config.get('NOISE_CLIP', 0.5)
        update_frequency = self.config.get('POLICY_FREQ', 2)
        reward_scale = self.config.get('REWARD_SCALE', 1.0)

        for step in range(episode_steps):

            # PSEUDO CODE: Sample mini-batch of N transitions from B
            # TODO: implement named tuple
            states, actions, rewards, next_states, dones = memory.sample(
                batch_size)

            # create Tensors from arrays
            state = torch.tensor(states, dtype=torch.float).to(device)
            action = torch.tensor(actions, dtype=torch.float).to(device)
            reward = torch.tensor(rewards, dtype=torch.float).reshape(
                (batch_size, 1)).to(device)
            next_state = torch.tensor(next_states, dtype=torch.float).to(device)
            done = torch.tensor(dones, dtype=torch.float).reshape(
                (batch_size, 1)).to(device)

            # PSEUDO CODE:
            #   select next action following the policy + clipped noise
            #   => calculate a_hat
            noise = torch.tensor(actions, dtype=torch.float).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.action_high, self.action_high)

            # PSEUDO CODE:
            # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
            #       = r                       otherwise
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward_scale * reward + ((1. - done) * discount * target_Q).detach()

            # PSEUDO CODE: Update critics
            # Get current Q estimates
            current_Q1 = self.critic_1(state, action)
            current_Q2 = self.critic_2(state, action)

            # Compute critic loss
            #   mse => element-wise mean squared error
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)

            # Update Critic # 1
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()

            # Update Critic # 2
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()

            # Delayed policy updates
            if step % update_frequency == 0:

                # Compute actor loss using critic # 1
                actor_loss = -self.critic_1(state, self.actor(state)).mean()

                # Optimize actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update target models
                # actor
                for parameter, target_parameter in zip(self.actor.parameters(),
                                                       self.actor_target.parameters()):
                    target_parameter.data.copy_(tau * parameter.data + (
                                1. - tau) * target_parameter.data)

                # critic # 1
                for parameter, target_parameter in zip(
                        self.critic_1.parameters(),
                        self.critic_1_target.parameters()):
                    target_parameter.data.copy_(tau * parameter.data + (
                                1. - tau) * target_parameter.data)

                # critic # 2
                for parameter, target_parameter in zip(
                        self.critic_2.parameters(),
                        self.critic_2_target.parameters()):
                    target_parameter.data.copy_(tau * parameter.data + (
                                1. - tau) * target_parameter.data)
