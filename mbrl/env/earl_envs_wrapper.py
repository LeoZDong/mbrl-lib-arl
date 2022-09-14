from gym import Wrapper


class TerminateOnSuccessWrapper(Wrapper):
    def __init__(self, env):
        assert hasattr(
            env, "is_successful"
        ), "Cannot apply terminate on success wrapper on non-EARL environments!"

        super().__init__(env)
        self.env = env

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        # Change termination condition
        done = done or self.env.is_successful(next_obs)
        return next_obs, reward, done, info
