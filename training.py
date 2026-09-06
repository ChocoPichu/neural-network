import pygame
import random
import math
import nnet
import config

class Runner:
    def __init__(self, pos):
        self.pos = pygame.math.Vector2(pos)
        self.dir = pygame.math.Vector2(-1, 0)
        self.brain = nnet.SimpleNeuralNetwork(30)
        self.fitness = 0
        self.alive = True

    def reset(self, new_pos):
        # Resets position and state for a new generation while KEEPING the brain.
        self.pos = pygame.math.Vector2(new_pos)
        self.dir = pygame.math.Vector2(-1, 0)
        self.fitness = 0
        self.alive = True

    def update(self, speed_output, turn_output, map_mask):
        if not self.alive:
            return

        self.fitness += 1

        # 2. Turn direction
        self.dir.rotate_ip(turn_output * config.turn_speed)

        # 3. Calculate target position
        next_pos = self.pos + self.dir * (speed_output * config.runner_speed)
        nx, ny = int(next_pos.x), int(next_pos.y)

        # 4. Check wall collision
        if 0 <= nx < config.WIDTH and 0 <= ny < config.HEIGHT:
            if not map_mask.get_at((nx, ny)):
                self.pos = next_pos

        # 5. Screen boundary limits
        self.pos.x = max(10, min(config.WIDTH - 10, self.pos.x))
        self.pos.y = max(10, min(config.HEIGHT - 10, self.pos.y))

class Chaser:
    def __init__(self, pos):
        self.pos = pygame.math.Vector2(pos)
        self.dir = pygame.math.Vector2(1, 0)
        self.brain = nnet.SimpleNeuralNetwork(30 + config.CHASER_SENSE_INPUTS)
        self.fitness = 0
        self.last_dist = None

    def reset(self, new_pos):
        self.pos = pygame.math.Vector2(new_pos)
        self.dir = pygame.math.Vector2(1, 0)
        self.fitness = 0
        self.last_dist = None