import copy

class Curriculum_Manager:
    def __init__(self, args):
        self.args = args
        self.t_max = self.args.t_max
        self.capability_config = self.args.env_args["capability_config"]
        
        self.C_menu = self.args.curriculum_menu # ["5v5", "10v10", "20v20"]

        for item in self.C_menu:
            assert isinstance(item, str), f"Invalid type: {item} (Expected string)"
            assert 'v' in item, f"Invalid format: {item} (Missing 'v')"
            left, right = item.split('v', 1)
            assert left.isdigit() and right.isdigit(), f"Invalid format: {item} (Non-numeric values)"
        print(f"All curriculum menu items are valid : {self.C_menu}")

        self.num_C = len(self.C_menu)
        self.C_interval = [ int((self.t_max / self.num_C) * i) for i in range(1, self.num_C +1)]
        self.current_C_level = 0
    
    def init_train_args(self, args):
        train_args = copy.deepcopy(args)
        train_args.env_args["capability_config"]["n_units"] = int(self.C_menu[0].split("v")[0])   
        train_args.env_args["capability_config"]["n_enemies"] = int(self.C_menu[0].split("v")[1]) # enemies
        return train_args
        
    def init_eval_args(self, eval_args):
        eval_args = copy.deepcopy(eval_args)
        eval_args.env_args["capability_config"]["n_units"] = int(self.C_menu[-1].split("v")[0])
        eval_args.env_args["capability_config"]["n_enemies"] = int(self.C_menu[-1].split("v")[1]) # enemies
        return eval_args
        
    def update(self, args, current_t_env):
            
        if current_t_env > self.C_interval[self.current_C_level]:
            self.current_C_level += 1
            args.env_args["capability_config"]["n_units"] = int(self.C_menu[self.current_C_level].split("v")[0]) # agents
            args.env_args["capability_config"]["n_enemies"] = int(self.C_menu[self.current_C_level].split("v")[1]) # enemies
            return args, True
        else:
            return args, False
            

                    
        
    