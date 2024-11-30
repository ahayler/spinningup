import argparse

from spinup.algos.reimplemented.utils import setup_logger_kwargs, Logger, get_act_dim
from spinup.algos.reimplemented.vpg.vpg import generate_trajectories
from spinup.algos.reimplemented.core import MLPActor, MLPValueFunction
import gymnasium as gym
import torch
import numpy as np
from tqdm import tqdm
import time

def ppo(
        env_fn,
        logger_kwargs: dict,
        hidden_size: int,
        num_hidden_layers: int,
        seed: int=None,
        num_epochs: int=100,
        steps_per_epoch: int=1000,
        gamma: float=0.99,
        lam: float=0.97,
        train_pol_iters: int=80,
        train_val_fn_iters: int = 80,
        pol_lr: float=3e-4,
        val_lr: float = 1e-3,
        target_kl: float=0.01,
        clip_ratio: float=0.2,
):
    """
    An implementation of PPO Clip using generalized advantage estimation and early stopping based on KL-divergence.

    Args:
        env_fn: Function to create gymnasium environment.
        logger_kwargs: Keyword arguments to pass to the logger.
        hidden_size: Size of the hidden layers.
        num_hidden_layers: Number of hidden layers.
        seed: Optional random seed to make experiments reproducible.
        num_epochs: Number of epochs to train for.
        steps_per_epoch: Number of steps per epoch.
        gamma: Discount factor.
        lam: Lambda hyperparameter (used for generalized advantage estimation).
        train_pol_iters: Maximum number of training iterations for the policy network per epoch.
        train_val_fn_iters: Number of training iterations for the value function network per epoch.
        val_lr: Learning rate for the value function.
        pol_lr: Learning rate for the policy network.
        target_kl: Target KL-divergence, if we achieve more 150% in an episode, we stop early.
        clip_ratio:
            The clip ratio controls until when we reward changes (in the right direction) of the ratio
            of probabilities between the new and old policy. You can still go further, it will just not be rewarded.
            This is normally noted with epsilon.
    """
    start_time = time.time()

    # Set up env
    env = env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = get_act_dim(env.action_space)

    # Setup the random seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        env.reset(seed=seed) # The seed gets passed the FIRST time the env gets reset and then never again

    # Setup the logger
    logger = Logger(**logger_kwargs)

    # Set up actor and value function
    actor = MLPActor(hidden_sizes=[hidden_size] * num_hidden_layers, action_space=env.action_space, obs_dim=obs_dim)
    val_fn = MLPValueFunction(hidden_sizes=[hidden_size] * num_hidden_layers, obs_dim=obs_dim)

    act_optim = torch.optim.Adam(actor.parameters(), lr=pol_lr)
    val_optim = torch.optim.Adam(val_fn.parameters(), lr=val_lr)

    for epoch in tqdm(range(num_epochs)):
        _res = generate_trajectories(
            env=env,
            steps_per_epoch=steps_per_epoch,
            actor=actor,
            gamma=gamma,
            lam=lam,
            obs_dim=obs_dim,
            act_dim=act_dim,
            value_func=val_fn,
        )

        observations = _res['observations']
        actions = _res['actions']
        log_p_old = _res['log_probs']
        rewards = _res['rewards']
        advantages = _res['advantages']

        # We first train the actor
        for pol_iter in range(train_pol_iters):
            act_optim.zero_grad()
            log_p = actor.get_log_prob_from_action(observations, actions)
            assert log_p.shape == log_p_old.shape
            ratio = torch.exp(log_p - log_p_old)
            clamped_adv = torch.clamp(ratio,1 - clip_ratio, 1 + clip_ratio) * advantages
            loss_act = -torch.min(ratio * advantages, clamped_adv).mean()

            # compute some extra info
            kl = (log_p - log_p_old).mean().item()
            entropy = actor.get_entropy(observations).mean().item()
            clamped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio) # This is not entirely true as we only clamp in the "good" direction
            clamped_ratio = torch.as_tensor(clamped, dtype=torch.float32).mean().item()


            # Implement early stopping
            if abs(kl) > 1.5 * target_kl:
                logger.info(f"Broke in epoch {epoch} after {pol_iter} iterations, because of high KL-divergence.")
                break

            loss_act.backward()
            act_optim.step()

        # act_optim.zero_grad()
        # log_p = actor.get_log_prob_from_action(observations, actions)
        # loss_act = - (log_p * advantages).mean()
        # loss_act.backward()
        # act_optim.step()


        # Then we train the value function
        val_loss_list = []
        for val_iter in range(train_val_fn_iters):
            val_optim.zero_grad()
            val_pred = val_fn(observations)

            val_loss = (val_pred - rewards).pow(2).mean()
            val_loss_list.append(val_loss.item())
            val_loss.backward()
            val_optim.step()

        episode_returns = np.array(_res['episode_returns'])
        traj_lengths = np.array(_res['traj_lengths'])
        val_loss_list = np.array(val_loss_list)

        if epoch % 10 == 0:
            elapsed_seconds = time.time() - start_time
            logger.info("")  # makes the logs more readable
            logger.info(f"Epoch: {epoch}, (Time: {int(elapsed_seconds // 60):}:{elapsed_seconds % 60:02.0f})")
            logger.info(f"Mean Rewards: {rewards.mean():.2f}, (Std: {rewards.std():.2f})")
            logger.info(f"Mean Episode Return: {episode_returns.mean():.2f} (Std: {episode_returns.std():.2f})")
            logger.info(f"Mean Estimated Advantage: {advantages.mean():.2f}, (Std: {advantages.std():.2f})")
            logger.info(f"Last KL-divergence: {kl:.2f}")
            logger.info(f"Last entropy: {entropy:.2f}")
            logger.info(f"Last clamped ratio: {clamped_ratio:.2f}")
            logger.info(f"Mean loss value function: {val_loss_list.mean():.2f}")
            logger.info(f"Last loss value function: {val_loss_list[-1]:.2f}")
            logger.info(f"")

            if len(traj_lengths) > 0:
                logger.info(f"Mean Trajectory Length: {traj_lengths.mean():.2f} (Std: {traj_lengths.std():.2f})")
            else:
                logger.info(f"Mean Trajectory Length undefined as trajectory lasted entire epoch.")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="HalfCheetah-v5")
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

    ppo(
        env_fn=(lambda: gym.make(args.env)),
        logger_kwargs=logger_kwargs,
        seed=args.seed,
        num_epochs=args.num_epochs,
        steps_per_epoch=args.steps_per_epoch,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        gamma=args.gamma,
        lam=args.lam,
    )