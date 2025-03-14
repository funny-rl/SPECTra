from functools import partial

from .multiagentenv import MultiAgentEnv
from .grf import Academy_3_vs_1_with_keeper, Run_pass_and_shoot_with_keeper, Pass_and_shoot_with_keeper

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {
    # "sc2": partial(env_fn, env=StarCraft2Env),
    # "matrix_game_1": partial(env_fn, env=Matrix_game1Env),
    # "matrix_game_2": partial(env_fn, env=Matrix_game2Env),
    # "matrix_game_3": partial(env_fn, env=Matrix_game3Env),
    # "mmdp_game_1": partial(env_fn, env=mmdp_game1Env)
    # "academy_3_vs_1_with_keeper": partial(env_fn, env=Academy_3_vs_1_with_Keeper),
    # "academy_pass_and_shoot_with_keeper": partial(env_fn, env=Academy_Pass_and_Shoot_with_Keeper),
    # "academy_run_pass_and_shoot_with_keeper": partial(env_fn, env=Academy_Run_Pass_and_Shoot_with_Keeper),
    "academy_3_vs_1_with_keeper" : partial(env_fn, env=Academy_3_vs_1_with_keeper),
    "run_pass_and_shoot_with_keeper" : partial(env_fn, env=Run_pass_and_shoot_with_keeper), 
    "pass_and_shoot_with_keeper" : partial(env_fn, env=Pass_and_shoot_with_keeper)
}
