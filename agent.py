import torch
import random
import numpy as np
from collections import deque # to store the memories
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000 # to store 100k items in memory
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0 # number of games
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate according to the formula
        self.memory = deque(maxlen=MAX_MEMORY) # popleft() once memory is exceeded it'll remove elements from left
        self.model = Linear_QNet(11, 256, 3) # 11 states, 3 numbers in action, can change hidden layers size as we saw in weka
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0] # taking the first position of head
        # points next to the head in all directions to check for dangers using the namedtuple
        point_l = Point(head.x - 20, head.y) # using 20 because of the block size taken initially
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        # current direction - only 1 value will be one
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or # if we're going right and theres danger in the next point in same direction
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or # if we're going up and theres danger in the next point in RIGHT direction
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location - checking food is in which direction compared to the snake head
            game.food.x < game.head.x,  # food left 
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        # end of the states

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done): # done is substitute for game over state. function to remember everything in the memory
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE: # comparing with the memory batch size
            mini_sample = random.sample(self.memory, BATCH_SIZE) # returns a list of tuples
        else:
            mini_sample = self.memory # taking the whole memory if not enough samples

        states, actions, rewards, next_states, dones = zip(*mini_sample) # putting together as separate zips of each kind of sample 
        self.trainer.train_step(states, actions, rewards, next_states, dones) # multiple states actions etc from all the past games. 
        
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    # used to train only for one game step and not the longer run


    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation of the environment during initial stages
        self.epsilon = 80 - self.n_games # more games = smaller epsilon i.e. decreased randomness. can change 80 to whatever number
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2) # random value b/w 0 and 2
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float) # converting state to a tensor of torch.float datatype
            prediction = self.model(state0) # a raw value from which we take the max value (refer to feed forward neural network)
            # taking action from the state
            move = torch.argmax(prediction).item() # tensor, can take only one item
            final_move[move] = 1

        return final_move


def train(): # using functions defined above to do the training 
    plot_scores = [] # list to keep track of scores
    plot_mean_scores = [] # orange line - mean of scores
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory (also called experience replay or replay memory) - used to train again based on all the previous moves, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()