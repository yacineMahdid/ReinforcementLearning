from ple.games.flappybird import FlappyBird
from ple import PLE

class Agent:

    def __init__(self, allowed_actions):
        self.allowed_actions = allowed_actions

    def pickAction(self,reward, observation):
        print(observation)
        return 119;

game = FlappyBird()
p = PLE(game, fps=30, display_screen=True)
agent = Agent(allowed_actions=p.getActionSet())

nb_frames = 1000
p.init()
reward = 0.0

for i in range(nb_frames):
   if p.game_over():
           p.reset_game()

   observation = p.getScreenRGB()
   action = agent.pickAction(reward, observation)
   reward = p.act(action)