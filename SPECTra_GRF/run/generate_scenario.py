import os 
import json
import random
import shutil
import atexit
import gfootball
import copy

import numpy as np
from run.scenario import *

def load_env(env_name, algo_name, unique_token, scenario_file_path):
    
    new_env_name = f"{env_name}_{algo_name}_{unique_token}.py"
    env_file_path = os.path.join(os.path.dirname(__file__), "scenario", f"{env_name}.py")
    if not os.path.exists(env_file_path):
        raise ImportError(f"Scenario module {env_name}.py not found in run/scenario/")
    
    new_file_path = f"{scenario_file_path}/{new_env_name}"    
    os.makedirs(scenario_file_path, exist_ok=True)
    shutil.copy(env_file_path, new_file_path)
    
    def del_scenario_file():
        if os.path.exists(new_file_path):
                os.remove(new_file_path)
    atexit.register(del_scenario_file)
    
    return new_env_name.split(".")[0]

class Scenario_Manager():
    def __init__(self, args):
        self.args = args
        self.new_env_name = load_env(
            env_name = self.args.env_args['env_name'],
            algo_name = self.args.name,
            unique_token = self.args.unique_token, 
            scenario_file_path = os.path.join(os.path.dirname(gfootball.__file__), "scenarios")
        )
        self.batch_size_run = self.args.batch_size_run
        
