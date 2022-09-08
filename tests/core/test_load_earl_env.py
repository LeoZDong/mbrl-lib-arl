import numpy as np
import pytest
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

import mbrl.models
import mbrl.util
import mbrl.util.common
import mbrl.util.env

initialize(config_path="test_resources")
cfg = compose(config_name="test_load_earl_envs.yaml")
print(OmegaConf.to_yaml(cfg))


@pytest.fixture(scope="class")
def prepare_env(request):
    cfg.overrides.env = "earl___tabletop_manipulation--train"
    (
        request.cls.train_env,
        request.cls.term_fn,
        request.cls.reward_fn,
    ) = mbrl.util.env.EnvHandler.make_env(cfg)

    cfg.overrides.env = "earl___tabletop_manipulation--eval"
    request.cls.eval_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg)

    # Create environment replay buffer for true rewards
    request.cls.train_env_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        request.cls.train_env.observation_space.shape,
        request.cls.train_env.action_space.shape,
    )
    request.cls.eval_env_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        request.cls.eval_env.observation_space.shape,
        request.cls.eval_env.action_space.shape,
    )

    # Propagate environment replay buffer
    request.cls.train_collect_rewards = mbrl.util.common.rollout_agent_trajectories(
        env=request.cls.train_env,
        steps_or_trials_to_collect=1000,
        agent=mbrl.planning.RandomAgent(request.cls.train_env),
        agent_kwargs={},
        replay_buffer=request.cls.train_env_buffer,
    )
    request.cls.eval_collect_rewards = mbrl.util.common.rollout_agent_trajectories(
        env=request.cls.eval_env,
        steps_or_trials_to_collect=1000,
        agent=mbrl.planning.RandomAgent(request.cls.eval_env),
        agent_kwargs={},
        replay_buffer=request.cls.eval_env_buffer,
    )


@pytest.mark.usefixtures("prepare_env")
class TestLoadEarlEnv:
    def test_train_eval_envs(self):
        """Test that eval env has terminations but train env does not."""
        # Train env should not have any dones
        assert not np.any(self.train_env_buffer.get_all().dones)

        # Eval env should have some dones
        assert np.any(self.eval_env_buffer.get_all().dones)

        # Eval horizon is 200, so there should be n_steps // 200 `dones`
        n_steps = len(self.eval_env_buffer.get_all().dones)
        assert self.eval_env_buffer.get_all().dones.sum() == n_steps // 200

    def test_true_reward(self):
        """Test that true reward is always 0."""
        assert np.all(np.array(self.train_collect_rewards) == 0)
        assert np.all(np.array(self.eval_collect_rewards) == 0)


class MockAgent:
    def __init__(self, length):
        self.actions = np.ones((length, 1))

    def plan(self, obs):
        return self.actions
