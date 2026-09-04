import pygame
import math
import random

pygame.init()

WIDTH = 800
HEIGHT = 600

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tag")
clock = pygame.time.Clock()

# Chaser position and directions
chaser_pos = pygame.math.Vector2(300, 400)
chaser_dir = pygame.math.Vector2(1, 0)
chaser_speed = 4
turn_speed = 3

# Setup (replace runner_x / runner_y)
runner_pos = pygame.math.Vector2(300, 200)
runner_dir = pygame.math.Vector2(-1, 0)  # Facing left initially
runner_speed = 4

# Pretending that these are 2 outputs from the AI brain
# values between -1 and 1, so I guess the AI uses the tanh graph to move positions.
# outputs from chaser ai
chaser_ai_output_x = 1
chaser_ai_output_y = -1
# outputs from runner ai
runner_ai_output_x = 1
runner_ai_output_y = -1

# Ai vision, radar style. So 8 lines coming out of the circle at 45 degree angles. maximum distance is 300 px, and current distance is 0
MAX_RADAR_DIST = 200
STEP_SIZE = 2 # Seems like if you do it 5 or more, it would go through the chaser and the runner, and not collide
NUM_RAYS = 8
current_dist = 0
radar_inputs = []
VISION_CONE = [-30, -15, -5, 0, 5, 15, 30]

# Ai vision, cone of vision style.
chaser_angle = 0
runner_angle = 180
turn_speed = 3

# Loading the first map
map_surface_1 = pygame.image.load("maps/map.png").convert_alpha()

# Create a mask of solid pixels, so everything which is not transparent becomes solid.
map_mask = pygame.mask.from_surface(map_surface_1)

# the game resetting function
chaser_score = 0
runner_score = 0
TIMER = 5 # for now 5 for testing purposes, until spawning in the map boxes is fixed.

font = pygame.font.SysFont("Arial", 20)

def reset_game():
    global chaser_pos, chaser_dir, runner_pos, runner_dir, start_time

    # Random spawn
    chaser_random_x = random.randint(1, WIDTH)
    chaser_random_y = random.randint(1, 500)

    runner_random_x = random.randint(1, WIDTH)
    runner_random_y = random.randint(1, 500)

    # Reset chaser
    chaser_pos = pygame.math.Vector2(chaser_random_x, chaser_random_y)
    chaser_dir = pygame.math.Vector2(1, 0)

    # Reset runner
    runner_pos = pygame.math.Vector2(runner_random_x, runner_random_y)
    runner_dir = pygame.math.Vector2(-1, 0)

    start_time = pygame.time.get_ticks()

# resetting the game at the start
start_time = pygame.time.get_ticks()
reset_game()

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Rotate direction vector
    chaser_dir.rotate_ip(chaser_ai_output_y * turn_speed)
    # AI movement logic for chaser
    # Calculate where the chaser wants to move
    chaser_next_pos = chaser_pos + chaser_dir * (chaser_ai_output_x * chaser_speed)
    cnx = int(chaser_next_pos.x)
    cny = int(chaser_next_pos.y)
    # Only move if the next pixel isn't inside a solid wall
    if 0 <= cnx < WIDTH and 0 <= cny < HEIGHT:
        if not map_mask.get_at((cnx, cny)):
            chaser_pos = chaser_next_pos
    # keep within boundaries
    chaser_pos.x = max(10, min(WIDTH - 10, chaser_pos.x))
    chaser_pos.y = max(10, min(WIDTH - 10, chaser_pos.y))

    # Rotate direction vector
    runner_dir.rotate_ip(runner_ai_output_y * turn_speed)
    # Runner AI movement logic
    # Calculate where the runner wants to move
    runner_next_pos = runner_pos + runner_dir * (runner_ai_output_x * runner_speed)
    rnx = int(runner_next_pos.x)
    rny = int(runner_next_pos.y)
    # Only move if the next pixel isn't inside a solid wall
    if 0 <= cnx < WIDTH and 0 <= cny < HEIGHT:
        if not map_mask.get_at((rnx, rny)):
            runner_pos = runner_next_pos
    runner_pos.x = max(10, min(WIDTH - 10, runner_pos.x))
    runner_pos.y = max(10, min(HEIGHT - 10, runner_pos.y))

    # Drawing part
    screen.fill('White')
    screen.blit(map_surface_1, (0, 0))

    # Win condition, if the chaser manages to touch the runner. Or runner survives the 20 seconds duration
    elapsed_seconds = (pygame.time.get_ticks() - start_time) / 1000
    time_remaining = max(0, int(TIMER - elapsed_seconds))

    if chaser_pos.distance_to(runner_pos) < 20:
        chaser_score += 1
        print(f"Chaser won. Score, Runner: {runner_score}. Chaser {chaser_score}")
        reset_game()

    elif time_remaining <= 0:
        runner_score += 1
        print(f"Runner survived. Score, Runner: {runner_score}. Chaser {chaser_score}")
        reset_game()

    # Chaser Vision Loop
    for i in range(NUM_RAYS):
        # Rotate ray relative to where the chaser is currently facing
        ray_dir = chaser_dir.rotate(i * 45)
        chaser_ray_pos = chaser_pos.copy()
        while current_dist < MAX_RADAR_DIST:
            current_dist += STEP_SIZE
            chaser_ray_pos = chaser_pos + (ray_dir * current_dist)
            # convert ray vector cords into integers, for mask lookup
            chaser_ray_x = int(chaser_ray_pos.x)
            chaser_ray_y = int(chaser_ray_pos.y)
            # Wall collision check
            if 0 <= chaser_ray_x < WIDTH and 0 <= chaser_ray_y < HEIGHT:
                if map_mask.get_at((chaser_ray_x, chaser_ray_y)):
                    break # hit a wall in the picture
            else:
                break # hit the screen border
            # Distance check to runner using .distance_to()
            if chaser_ray_pos.distance_to(runner_pos) < 10:
                break
        pygame.draw.line(screen, (0, 255, 0), chaser_pos, chaser_ray_pos)
        current_dist = 0

    for angle in VISION_CONE:
        # Rotate relative to chaser_dir heading
        ray_dir = chaser_dir.rotate(angle)
        chaser_ray_pos = chaser_pos.copy()
        while current_dist < MAX_RADAR_DIST:
            current_dist += STEP_SIZE
            chaser_ray_pos = chaser_pos + (ray_dir * current_dist)
            chaser_ray_x = int(chaser_ray_pos.x)
            chaser_ray_y = int(chaser_ray_pos.y)
            # Wall collision check
            if 0 <= chaser_ray_x < WIDTH and 0 <= chaser_ray_y < HEIGHT:
                if map_mask.get_at((chaser_ray_x, chaser_ray_y)):
                    break  # hit a wall in the picture
            if chaser_ray_pos.distance_to(runner_pos) < 10:
                break
        pygame.draw.line(screen, (255, 0, 255), chaser_pos, chaser_ray_pos)
        current_dist = 0

    # Vision loop, and drawing rays for runner
    for i in range(NUM_RAYS):
        # Rotate ray relative to where the chaser is currently facing
        ray_dir = runner_dir.rotate(i * 45)
        runner_ray_pos = runner_pos.copy()
        while current_dist < MAX_RADAR_DIST:
            current_dist += STEP_SIZE
            runner_ray_pos = runner_pos + (ray_dir * current_dist)
            # convert ray vector cords into integers, for mask lookup
            runner_ray_x = int(runner_ray_pos.x)
            runner_ray_y = int(runner_ray_pos.y)
            # Wall collision check
            if 0 <= runner_ray_x < WIDTH and 0 <= runner_ray_y < HEIGHT:
                if map_mask.get_at((runner_ray_x, runner_ray_y)):
                    break # hit a wall in the picture
            else:
                break # hit the screen border
            # Distance check to runner using .distance_to()
            if runner_ray_pos.distance_to(chaser_pos) < 10:
                break
        pygame.draw.line(screen, (0, 255, 0), runner_pos, runner_ray_pos)
        current_dist = 0

    for angle in VISION_CONE:
        # Rotate relative to runner_dir heading
        ray_dir = runner_dir.rotate(angle)
        runner_ray_pos = runner_pos.copy()

        while current_dist < MAX_RADAR_DIST:
            current_dist += STEP_SIZE
            runner_ray_pos = runner_pos + (ray_dir * current_dist)
            # convert ray vector cords into integers, for mask lookup
            runner_ray_x = int(runner_ray_pos.x)
            runner_ray_y = int(runner_ray_pos.y)
            # Wall collision check
            if 0 <= runner_ray_x < WIDTH and 0 <= runner_ray_y < HEIGHT:
                if map_mask.get_at((runner_ray_x, runner_ray_y)):
                    break  # hit a wall in the picture
            if runner_ray_pos.distance_to(chaser_pos) < 10:
                break
        pygame.draw.line(screen, (255, 0, 255), runner_pos, runner_ray_pos)
        current_dist = 0

    # Chaser
    pygame.draw.circle(screen, (255, 0, 0), chaser_pos, 10)
    # Runner
    pygame.draw.circle(screen, (0, 0, 255), runner_pos, 10)

    # timer and scoreboard
    timer_text = font.render(f"Time: {time_remaining}s", True, (0, 0, 0))
    score_text = font.render(f"Chaser: {chaser_score}  |  Runner: {runner_score}", True, (0, 0, 0))

    screen.blit(timer_text, (WIDTH // 2 - timer_text.get_width() // 2, 10))
    score_rect = score_text.get_rect(topright=(WIDTH - 10, 10))
    screen.blit(score_text, score_rect)

    # update the screen
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
