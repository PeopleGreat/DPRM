# --------------------------------------------------------
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import copy
import json
import math
import os
# os.environ['MUJOCO_GL'] = 'egl'
import random
import sys
import time

import dmc2gym
import gym
import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from torchvision import transforms

import utils
from logger import Logger
from mtm_sac import MTMSacAgent
from video import VideoRecorder


def evaluate(env, mode, agent, video, num_episodes, L, step, args, test_env=False):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        #prefix = 'stochastic_' if sample_stochastically else ''
        for i in range(num_episodes):
            obs = env.reset()

            video.init(enabled=True)
            done = False
            episode_reward = 0
            while not done:
                # center crop image
                if args.encoder_type == 'pixel':
                    obs = utils.center_crop_image(obs, args.image_size)
                # with utils.eval_mode(agent):
                #     if sample_stochastically:
                #         action = agent.sample_action(obs)
                #     else:
                action = agent.select_action(obs)
                obs, reward, done, _ = env.step(action)
                video.record(env)
                episode_reward += reward
            video.save('%d.mp4' % i)
            L.log('eval/' + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)

        L.log('eval/' + 'eval_time', time.time() - start_time, step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        L.log('eval/' + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + 'best_episode_reward', best_ep_reward, step)

    run_eval_loop(sample_stochastically=False)
    L.dump(step)
    print(all_ep_rewards, np.mean(all_ep_rewards))


def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'mtm_sac':
        return MTMSacAgent(
            predict_type = args.predict_type,
            mask_img = args.mask_img,
            mask_feature = args.mask_feature,
            imgspatial = args.imgspatial,
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            augmentation=args.augmentation,
            transition_model_type=args.transition_model_type,
            transition_model_layer_width=args.transition_model_layer_width,
            jumps=args.jumps,
            latent_dim=args.latent_dim,
            num_aug_actions=args.num_aug_actions,
            loss_space=args.loss_space,
            bp_mode=args.bp_mode,
            cycle_steps=args.cycle_steps,
            cycle_mode=args.cycle_mode,
            fp_loss_weight=args.fp_loss_weight,
            bp_loss_weight=args.bp_loss_weight,
            rc_loss_weight=args.rc_loss_weight,
            vc_loss_weight=args.vc_loss_weight,
            reward_loss_weight=args.reward_loss_weight,
            time_offset=args.time_offset,
            momentum_tau=args.momentum_tau,
            aug_prob=args.aug_prob,
            auxiliary_task_lr=args.auxiliary_task_lr,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            curl_latent_dim=args.curl_latent_dim,
            sigma=args.sigma,
            mask_ratio=args.mask_ratio,
            patch_size=args.patch_size,
            block_size=args.block_size,
            num_attn_layers=args.num_attn_layers)
    else:
        assert 'agent is not supported: %s' % args.agent


@hydra.main(config_path="./configs", config_name="cheetah_run")
def main(args: DictConfig) -> None:
    args.log_interval *= args.action_repeat
    args.seed = args.seed or args.seed_and_gpuid[0]
    args.gpuid = args.gpuid or args.seed_and_gpuid[1]
    args.domain_name = args.domain_name or args.env_name.split('/')[0]
    args.task_name = args.task_name or args.env_name.split('/')[1]
    if args.seed == -1:
        args.seed = np.random.randint(1, 1000000)
    print(f'seed:{args.seed}')
    torch.cuda.set_device(args.gpuid)
    utils.set_seed_everywhere(args.seed)
    from env.wrappers import make_env
    mode = 'train'
    env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		# seed=args.seed+42,
        seed=args.seed,
		action_repeat=args.action_repeat,
		mode=mode,
		intensity=0.
	) #if args.eval_mode is not None else None
    #env.seed(args.seed)

    # stack several consecutive frames together
    # if args.encoder_type == 'pixel':
    #     env = utils.FrameStack(env, k=args.frame_stack)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    action_shape = env.action_space.shape

    if args.encoder_type == 'pixel':
        obs_shape = (3 * args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (3 * args.frame_stack,
                             args.pre_transform_image_size,
                             args.pre_transform_image_size)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    agent = make_agent(obs_shape=obs_shape,
                       action_shape=action_shape,
                       args=args,
                       device=device)

    load_model_dir = args.work_dir
    step = 100000
    agent.load(load_model_dir, step)
    
    index = load_model_dir.find('model')
    args.work_dir = load_model_dir[:index]
    args.work_dir = args.work_dir + f'eval_{mode}_{step}'
    print(args.work_dir)
    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))

    video = VideoRecorder(video_dir if args.save_video else None)
    L = Logger(args.work_dir, use_tb=args.save_tb, use_wandb=args.wandb)
    evaluate(env, mode, agent, video, args.num_eval_episodes, L, step, args)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
