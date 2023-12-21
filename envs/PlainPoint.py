import numpy as np

class action_space():
    def __init__(self, size):
        self.size = size
        
    def __get__(self):
        return self.size
    
    def sample(self):
        return np.random.rand(self.size)
    
    @property
    def low(self):
        return np.zeros((self.size,))
    
    @property
    def high(self):
        return np.ones((self.size,))

class plainPoint():
    def __init__(self, X_weight:int = 1, Y_weight:int = 1, step_limit=50, Displacement: bool=True):
        self.position = None
        self.steps = None
        self.action_space = action_space(2)
        self.X_weight = X_weight
        self.Y_weight = Y_weight
        self.Displacement = Displacement
        self.step_limit = step_limit
    
    def reset(self, seed=0):
        if self.Displacement:
            self.position = np.zeros((2,))
        else:
            self.position = np.ones((2,))
        self.steps = 0
        return (self.position, {})
    
    def step(self, action):
        """
        Args:
            action (np.array): Shape [2,]
        """
        if self.position is None or self.steps == self.step_limit:
            raise Exception("Environment not initialized.")
        else:
            if self.Displacement:
                self.position += action
            reward = self.X_weight*action[0] + self.Y_weight*action[1]
            self.steps += 1
            
            if self.steps == self.step_limit:
                return self.position, reward, True, False, {}
            else:
                return self.position, reward, False, False, {}
        
    def render(self):
        return np.zeros((1,))
    
    @property
    def data(self):
        return np.zeros((1,))
            
class plainPointMAX(plainPoint):
    def __init__(self, X_weight: int = 1, Y_weight: int = 1, step_limit=1, Displacement: bool = False):
        super().__init__(X_weight, Y_weight, step_limit, Displacement)
        
    def step(self, action):
        if self.position is None or self.steps == self.step_limit:
            raise Exception("Environment not initialized.")
        else:
            if self.Displacement:
                self.position += action
            reward = min(action[0]+action[1], 1.5) - 0.5*(action[0]+action[1])
            self.steps += 1
            
            if self.steps == self.step_limit:
                return self.position, reward, True, False, {}
            else:
                return self.position, reward, False, False, {}
            
        
        