# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import omegaconf
import torch

import mbrl.algorithms.mbpo as mbpo
import mbrl.algorithms.pets as pets
import mbrl.algorithms.planet as planet
import mbrl.util.env


@hydra.main(config_path="conf", config_name="main")
def run(cfg: omegaconf.DictConfig):
    env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(cfg)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.algorithm.name == "pets":
        return pets.train(env, term_fn, reward_fn, cfg)
    if cfg.algorithm.name == "mbpo":
        # When creating the eval env, change the stage tag to be 'eval' in the string
        # description of the environment.
        if "env" in cfg.overrides.keys() and "earl___" in cfg.overrides.env:
            env_type, name = cfg.overrides.env.split("___")
            env_name, _ = name.split("--")
            cfg.overrides.env = "___".join([env_type, "--".join([env_name, "eval"])])

        test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg)

        # Optionally use true reward function in model rollout
        if cfg.overrides.get("learned_rewards", True):
            return mbpo.train(env, test_env, term_fn, cfg)
        else:
            return mbpo.train(
                env, test_env, term_fn, cfg, reward_fn_in_rollout=reward_fn
            )
    if cfg.algorithm.name == "planet":
        return planet.train(env, cfg)


if __name__ == "__main__":
    run()
