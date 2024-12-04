import argparse
from tabnanny import check

import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm

from spinup.algos.reimplemented.utils import setup_logger_kwargs, Logger, _to_tensor
from spinup.algos.reimplemented.off_policy.core import CategoricalMLPQNet


class ReplayBuffer:
    def __init__(self, capacity: int, act_dim: int, obs_dim: int):
        assert capacity > 0
        # The +1 is needed because this implementation of the buffer only can store up to n-1 elements in an n element buffer
        self.state_from = torch.zeros(capacity + 1, obs_dim)
        self.state_to = torch.zeros(capacity + 1, obs_dim)
        self.rewards = torch.zeros(capacity + 1)
        self.actions = torch.zeros(capacity + 1, act_dim, dtype=torch.int64)
        self.terminals = torch.zeros(capacity + 1, dtype=torch.bool)
        self.capacity = capacity + 1
        self.start_ptr = 0
        self.end_ptr = 0

    def add(self, state_from: torch.tensor, state_to: torch.tensor, reward: float, action: torch.tensor, terminal: bool):
        """
        Add element to buffer. Starts overwriting elements when they are needed.
        """
        self.state_from[self.end_ptr] = _to_tensor(state_from)
        self.state_to[self.end_ptr] = _to_tensor(state_to)
        self.rewards[self.end_ptr] = _to_tensor(reward)
        self.actions[self.end_ptr] = _to_tensor(action)
        self.terminals[self.end_ptr] = _to_tensor(terminal)
        self.actions[self.end_ptr] = _to_tensor(action)

        self.end_ptr = (self.end_ptr + 1) % self.capacity
        if self.start_ptr == self.end_ptr: self.start_ptr = (self.start_ptr + 1) % self.capacity

    def get_size(self):
        return (self.end_ptr - self.start_ptr) % self.capacity

    def sample(self, batch_size: int) -> dict:
        indices = torch.randint(self.start_ptr, self.start_ptr + self.get_size(), (batch_size,)) % self.capacity

        batch_dict = dict(
            state_from=self.state_from[indices],
            state_to=self.state_to[indices],
            actions=self.actions[indices],
            terminal=self.terminals[indices],
            rewards=self.rewards[indices]
        )

        return batch_dict


def dqn(
        env_fn,
        steps_per_epoch: int,
        num_epochs: int,
        logger_kwargs: dict,
        num_layers: int=2,
        hidden_size: int = 64,
        seed: int=None,
        batch_size: int=100,
        lr: float=1e-3,
        update_every: int=50,
        burn_in: int=1000,
        buffer_size: int = 10000,
        epsilon: float=0.1,
        fade_out_duration: int=10e4,
        gamma: float=0.99,
        eval_steps: int=10000,
        eval_every_epochs: int=5,
):
    """
    A basic implementation of Deep Q-Networks for discrete sample spaces.

    Args:
        env_fn: Function to create gym environment.
        steps_per_epoch: Number of steps in one epoch.
        num_epochs: Number of epochs.
        logger_kwargs: Keyword arguments to pass to the logger.
        num_layers: Number of layers in the Q-Net.
        hidden_size: The size of each hidden layer.
        seed: Random seed.
        batch_size: The size of each mini-batch.
        lr: The learning rate.
        update_every: The number of steps between updates.
        burn_in: The number of steps for which no update should be performed.
        buffer_size: The size of the replay buffer.
        epsilon: The (base) exploration rate.
        fade_out_duration: The number of steps we take until we reduced the exploration rate from 1 to 0.1.
        gamma: The discount factor.
        eval_steps: The number of steps to run each evaluation for.
        eval_every_epochs: The number of epochs between evaluations runs.
    """
    # Create env
    env = env_fn()

    # Create logger
    logger = Logger(**logger_kwargs)

    # Compute the decrease_rate
    def get_exploration_rate(step: int):
        return max(epsilon, 1 - step * (1 - epsilon) / fade_out_duration)

    # Function to evaluate the model
    def run_eval(model: CategoricalMLPQNet):
        # Run evaluation
        with torch.no_grad():
            env_eval = env_fn()
            obs = env_eval.reset(seed=10)[0]

            traj_lens = []
            traj_rwds = []

            traj_reward = 0
            start_index = 0

            for step in range(eval_steps):
                act = model(obs).argmax().item()
                obs, rew, d, _, _ = env_eval.step(act)

                traj_reward += reward

                if d:
                    traj_lens.append(step - start_index)
                    start_index = step
                    traj_rwds.append(traj_reward)
                    traj_reward = 0

                    obs = env_eval.reset()[0]

        return dict(traj_lengths=np.array(traj_lens), traj_rewards=np.array(traj_rwds))



    # set the random seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        env.reset(seed=seed)

    # Get initial observation
    observation = env.reset()[0]

    # Set up the Q-Network and optimizer
    obs_dim = env.observation_space.shape[0]
    act_dim = 1 # We are currently in the discrete case
    n_possible_actions = env.action_space.n
    q_net = CategoricalMLPQNet(hidden_sizes=num_layers * [hidden_size], obs_dim=obs_dim, n_possible_actions=n_possible_actions)
    optim = torch.optim.Adam(q_net.parameters(), lr=lr)

    # Set up the replay buffer
    buffer = ReplayBuffer(capacity=buffer_size, act_dim=act_dim, obs_dim=obs_dim)

    step_count = 0

    for epoch in tqdm(range(num_epochs)):
        for _ in range(steps_per_epoch):
            # Search requires no gradients
            with torch.no_grad():
                exploration_rate = get_exploration_rate(step_count)

                # sample an action
                if np.random.rand() < exploration_rate:
                    # Choose random action
                    action = env.action_space.sample()
                else:
                    # Choose the action that the Q-Function deems best
                    action = torch.argmax(q_net(observation)).item()

                # step
                next_observation, reward, done, _, _ = env.step(action)

                # add it to the buffer
                buffer.add(state_from=observation, state_to=next_observation, reward=reward, action=action, terminal=done)

                # set the nex observation
                if not done:
                    observation = next_observation
                else:
                    observation = env.reset()[0]

            # Determine whether we need to update or not
            if (step_count > burn_in) and (step_count % update_every == 0):
                # update our Q-Net
                batch = buffer.sample(batch_size)

                states_from = batch['state_from']
                states_to = batch['state_to']
                actions = batch['actions']
                rewards = batch['rewards']
                terminals = _to_tensor(batch['terminal']) # Convert the boolean tensor to float

                #print(step_count)

                # get the y's
                with torch.no_grad():
                    val_next = q_net(states_to).detach().max(1)[0]
                    y = gamma * (1 - terminals) * val_next + rewards # TODO: Breakpoint in here later to make sure this is working as intended

                val = q_net(states_from).gather(1, actions)

                # step using the optimizer
                optim.zero_grad()
                loss = (val.squeeze(-1) - y).pow(2).mean()
                #print(loss)
                loss.backward()
                optim.step()

            step_count += 1

        if epoch % eval_every_epochs == 0:
            eval_results = run_eval(q_net)
            traj_lengths = eval_results['traj_lengths']
            traj_rewards = eval_results['traj_rewards']

            logger.info(f"")
            if len(traj_lengths) > 0:
                logger.info(f"Mean trajectory length: {traj_lengths.mean():.2f}, (Std {traj_lengths.std():.2f})")
                logger.info(f"Mean trajectory reward: {traj_rewards.mean():.2f}, (Std {traj_rewards.std():.2f})")
            else:
                logger.info(f"Episode ran for the entire evaluation.")
            logger.info(f"Current exploration percentage: {get_exploration_rate(step_count)*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--env', type=str, default='LunarLander-v3')
    parser.add_argument("--exp", type=str, default='test')
    parser.add_argument('--steps_per_epoch', type=int, default=10000)
    parser.add_argument('--num_epochs', type=int, default=100)
    args = parser.parse_args()

    logger_kwargs = setup_logger_kwargs(args.exp, args.seed)
    dqn(
        env_fn=lambda: gym.make(args.env),
        logger_kwargs=logger_kwargs,
        seed=args.seed,
        steps_per_epoch=args.steps_per_epoch,
        num_epochs=args.num_epochs,
    )