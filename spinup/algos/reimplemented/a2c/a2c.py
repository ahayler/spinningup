from spinup.algos.reimplemented.utils import setup_logger_kwargs, Logger, get_act_dim
import gymnasium as gym
import argparse
import numpy as np
import torch
import time
from tqdm import tqdm

from spinup.algos.reimplemented.core import A2CActor
from spinup.algos.reimplemented.vpg.vpg import generate_trajectories


def a2c(
        env_fn,
        hidden_size: int,
        num_hidden_layers: int,
        steps_per_epoch: int, num_epochs: int,
        gamma: float,
        lam: float,
        lr: float=5e-4,
        actor_loss_coef: float=1,
        critic_loss_coef: float=0.5,
        entropy_loss_coef: float=0.01,
        seed: int=None,
        logger_kwargs=None,
):
    """
    An implementation of A2C with normalized generalized advantage estimation.

    Args:
        env_fn: A function that returns a gym environment.
        hidden_size: The size of a hidden layer.
        num_hidden_layers: The number of hidden layers.
        steps_per_epoch: The number of steps per epoch.
        num_epochs: The number of epochs.
        gamma: The discount factor (used for rewards-to-go).
        lam: The discount factor (used for generalized advantage estimation)
        lr: The general learning rate for the loss that combines actor loss, value function loss and entropy loss.
        actor_loss_coef: The loss coefficient for the actor loss.
        critic_loss_coef: The loss coefficient for the critic loss.
        entropy_loss_coef: The loss coefficient for the entropy loss.
        seed: The random seed used for numpy and torch.
        logger_kwargs: A dictionary of keyword arguments to pass to the logger.
    """

    env = env_fn()
    start_time = time.time()

    # Set the random states
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.action_space.seed(seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = get_act_dim(env.action_space)

    a2c_actor = A2CActor(
        hidden_sizes=[hidden_size] * num_hidden_layers,
        action_space=env.action_space,
        obs_dim=obs_dim,
    )

    # Setup logger
    logger = Logger(**logger_kwargs)
    logger.setup_pytorch_saver_elements(a2c_actor)

    optim = torch.optim.Adam(a2c_actor.parameters(), lr=lr)
    val_loss_fn = torch.nn.MSELoss()

    for epoch in tqdm(range(num_epochs)):
        _res = generate_trajectories(
            env=env,
            steps_per_epoch=steps_per_epoch,
            actor=a2c_actor,
            gamma=gamma,
            lam=lam,
            obs_dim=obs_dim,
            act_dim=act_dim,
            value_func=a2c_actor,
        )

        actions = _res["actions"]
        observations = _res["observations"]
        advantages = _res["advantages"]
        rewards = _res["rewards"]
        traj_lengths = _res["traj_lengths"]
        episode_returns = _res["episode_returns"]

        # for A2C we update all things at the same time using a joint loss function
        optim.zero_grad()

        # Actor loss
        log_probs = a2c_actor.get_log_prob_from_action(observations, actions)
        actor_pol = - (advantages.detach() * log_probs).mean() # advantages should already be detached

        # Value function loss
        vals = a2c_actor.get_value(observations)
        critic_loss = val_loss_fn(vals.squeeze(1), rewards)

        # Entropy loss
        entropy = a2c_actor.get_entropy(observations)
        entropy_loss = entropy.mean()

        loss = actor_loss_coef * actor_pol + critic_loss_coef * critic_loss + entropy_loss_coef * entropy_loss
        loss.backward()

        optim.step()


        episode_returns = np.array(episode_returns)

        if epoch % 10 == 0:
            elapsed_seconds = time.time() - start_time
            logger.info("")  # makes the logs more readable
            logger.info(f"Epoch: {epoch}, (Time: {int(elapsed_seconds // 60):}:{elapsed_seconds % 60:02.0f})")
            logger.info(f"Mean Rewards: {rewards.mean():.2f}, (Std: {rewards.std():.2f})")
            logger.info(f"Mean Episode Return: {episode_returns.mean():.2f} (Std: {episode_returns.std():.2f})")
            logger.info(f"Mean Estimated Advantage: {advantages.mean():.2f}, (Std: {advantages.std():.2f})")
            logger.info(f"Loss: {loss.item():.2f}")
            logger.info(f"Actor Loss: {actor_pol.item():.2f}")
            logger.info(f"Critic (Value Function) Loss: {critic_loss.item():.2f}")
            logger.info(f"Entropy Loss: {entropy_loss:.2f}, (Std: {entropy.std().item():.2f})")
            logger.info(f"")

            if len(traj_lengths) > 0:
                logger.info(f"Mean Trajectory Length: {traj_lengths.mean():.2f} (Std: {traj_lengths.std():.2f})")
            else:
                logger.info(f"Mean Trajectory Length undefined as trajectory lasted entire epoch.")

    # Save the final model
    logger.save_state({"env": env}, iteration=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--steps_per_epoch", type=int, default=4000)
    parser.add_argument("--num_epochs", type=int, default=800)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", type=str, default="test")
    args = parser.parse_args()

    # set up logger kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    a2c(
        env_fn=(lambda: gym.make(args.env)),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        steps_per_epoch=args.steps_per_epoch,
        num_epochs=args.num_epochs,
        gamma=args.gamma,
        lam=args.lam,
        seed=args.seed,
        logger_kwargs=logger_kwargs,
    )