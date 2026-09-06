import pygame

pygame.init()

WIDTH = 800
HEIGHT = 600

# setup parameters
MAX_RADAR_DIST = 400
STEP_SIZE = 2
NUM_RAYS = 8
VISION_CONE = [-30, -15, -5, 0, 5, 15, 30]

chaser_speed = 6
runner_speed = 4
turn_speed = 3
chaser_turn_speed = 4

# Spawning: place the chaser within this ring around its runner so every round
# starts as a feasible chase instead of across the map.
SPAWN_MIN_DIST = 60
SPAWN_MAX_DIST = 250

# Tag reward: base bonus + extra for tagging early (early_bonus * seconds left).
TAG_BASE_REWARD = 300
TAG_EARLY_BONUS_PER_SEC = 20

# Dense pursuit reward: bonus per pixel the chaser closed on the runner in one tick.
CLOSE_PROGRESS_REWARD = 2.0

# Extra homing-sense inputs appended to the chaser's ray inputs (dx, dy, dist).
CHASER_SENSE_INPUTS = 3

# Evolution
MUTATION_RATE = 0.15
MUTATION_STRENGTH = 0.3

TIMER = 15
POPULATION_SIZE = 20
generation = 1

font = pygame.font.SysFont("Arial", 20)

