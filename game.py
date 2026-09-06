import pygame
import random
import copy
import pickle
import os
import glob
import nnet
import config
from training import Runner, Chaser

# Startup console menu
print("====================================")
print("       TAG AI - MODE SELECT         ")
print("====================================")
print("1. Training Mode (50 parallel pairs, auto-save every 5 gens)")
print("2. 1v1 Mode (20s round, load trained models, no evolution)")
mode_choice = input("Select mode (1 or 2): ").strip()

is_training_mode = mode_choice != "2"
population_size = 50 if is_training_mode else 1
round_timer = 10 if is_training_mode else 20

SIDEBAR_WIDTH = 220

# Load all playable maps (maps/1.png, 2.png, ...) and cycle through them each round.
map_paths = sorted(glob.glob(os.path.join("maps", "*.png")),
                   key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
current_map_index = 0
map_surface = None
map_mask = None


def load_map(index):
    global map_surface, map_mask
    map_surface = pygame.image.load(map_paths[index % len(map_paths)]).convert_alpha()
    map_mask = pygame.mask.from_surface(map_surface)


pygame.init()
screen = pygame.display.set_mode((config.WIDTH + SIDEBAR_WIDTH, config.HEIGHT))
pygame.display.set_caption("Tag - AI Co-Evolution" if is_training_mode else "Tag - 1v1 Evaluation")
clock = pygame.time.Clock()

load_map(0)

# Win counters across all played rounds.
# Runner wins by surviving the full round; Chaser wins by tagging the runner.
round_stats = {"chaser_wins": 0, "runner_wins": 0}


def record_round_results():
    for r in runners:
        if getattr(r, 'alive', True):
            round_stats["runner_wins"] += 1
        else:
            round_stats["chaser_wins"] += 1


def get_random_valid_pos(padding=15):
    while True:
        x = random.randint(padding, config.WIDTH - padding)
        y = random.randint(padding, config.HEIGHT - padding)
        if not map_mask.get_at((x, y)):
            return pygame.math.Vector2(x, y)


def spawn_chaser_near(runner_pos):
    # Find a free spot in a ring around the runner so the chase is feasible.
    for _ in range(200):
        angle = random.uniform(0, 360)
        radius = random.uniform(config.SPAWN_MIN_DIST, config.SPAWN_MAX_DIST)
        candidate = runner_pos + pygame.math.Vector2(radius, 0).rotate(angle)
        cx, cy = int(candidate.x), int(candidate.y)
        if 0 < cx < config.WIDTH and 0 < cy < config.HEIGHT and not map_mask.get_at((cx, cy)):
            return candidate
    return get_random_valid_pos()


def spawn_pair(chaser, runner):
    # Runner spawns first, then the chaser is placed near it. The runner faces
    # away from the chaser and the chaser faces straight at the runner.
    r_pos = get_random_valid_pos()
    c_pos = spawn_chaser_near(r_pos)

    runner.pos = r_pos
    runner.dir = (r_pos - c_pos).normalize() if r_pos != c_pos else pygame.math.Vector2(-1, 0)
    runner.fitness = 0
    runner.alive = True

    chaser.pos = c_pos
    chaser.dir = (r_pos - c_pos).normalize() if r_pos != c_pos else pygame.math.Vector2(1, 0)
    chaser.fitness = 0
    chaser.last_dist = c_pos.distance_to(r_pos)


# create population
chasers = [Chaser(get_random_valid_pos()) for _ in range(population_size)]
runners = [Runner(get_random_valid_pos()) for _ in range(population_size)]
for i in range(population_size):
    spawn_pair(chasers[i], runners[i])
start_time = pygame.time.get_ticks()
total_trained_generations = 0


def save_best_agents(gen_count):
    best_runner = max(runners, key=lambda r: getattr(r, 'fitness', 0))
    best_chaser = max(chasers, key=lambda c: getattr(c, 'fitness', 0))

    with open("best_runner.pkl", "wb") as f:
        pickle.dump(best_runner.brain, f)
    with open("best_chaser.pkl", "wb") as f:
        pickle.dump(best_chaser.brain, f)
    with open("meta.pkl", "wb") as f:
        pickle.dump({"generations": gen_count}, f)
    print(f"[Auto-Save] Model weights saved at generation {gen_count}")


def load_best_agents():
    global total_trained_generations
    if os.path.exists("best_runner.pkl") and os.path.exists("best_chaser.pkl"):
        with open("best_runner.pkl", "rb") as f:
            saved_runner_brain = pickle.load(f)
        with open("best_chaser.pkl", "rb") as f:
            saved_chaser_brain = pickle.load(f)

        for r in runners:
            r.brain = copy.deepcopy(saved_runner_brain)
        for c in chasers:
            c.brain = copy.deepcopy(saved_chaser_brain)

        if os.path.exists("meta.pkl"):
            with open("meta.pkl", "rb") as f:
                meta = pickle.load(f)
                total_trained_generations = meta.get("generations", 0)
        print(f"Loaded trained models (Trained for {total_trained_generations} gens)")
    else:
        print("No saved models found! Running with fresh random weights.")


if not is_training_mode:
    load_best_agents()


def reset_round():
    global start_time, current_map_index
    current_map_index = (current_map_index + 1) % len(map_paths)
    load_map(current_map_index)
    for i in range(population_size):
        spawn_pair(chasers[i], runners[i])

    start_time = pygame.time.get_ticks()


def evolve_agents(population, agent_class):
    population.sort(key=lambda a: getattr(a, 'fitness', 0), reverse=True)
    num_elites = max(1, int(population_size * 0.2))
    elites = population[:num_elites]

    new_pop = []

    for elite in elites:
        clone = agent_class(get_random_valid_pos())
        clone.brain.w1 = copy.deepcopy(elite.brain.w1)
        clone.brain.b1 = copy.deepcopy(elite.brain.b1)
        clone.brain.w2 = copy.deepcopy(elite.brain.w2)
        clone.brain.b2 = copy.deepcopy(elite.brain.b2)
        new_pop.append(clone)

    while len(new_pop) < population_size:
        parent = random.choice(elites)
        child = agent_class(get_random_valid_pos())
        child.brain.w1 = copy.deepcopy(parent.brain.w1)
        child.brain.b1 = copy.deepcopy(parent.brain.b1)
        child.brain.w2 = copy.deepcopy(parent.brain.w2)
        child.brain.b2 = copy.deepcopy(parent.brain.b2)

        if hasattr(child.brain, 'mutate'):
            child.brain.mutate(mutation_rate=config.MUTATION_RATE, mutation_strength=config.MUTATION_STRENGTH)

        new_pop.append(child)

    return new_pop


def evolve_population():
    global chasers, runners, total_trained_generations
    runners = evolve_agents(runners, Runner)
    chasers = evolve_agents(chasers, Chaser)
    config.generation += 1
    total_trained_generations = config.generation

    # Auto-save every 5 generations
    if config.generation % 5 == 0:
        save_best_agents(config.generation)

    reset_round()


def draw_sidebar():
    # Non-playable stats panel on the left side of the window.
    pygame.draw.rect(screen, (45, 45, 55), (0, 0, SIDEBAR_WIDTH, config.HEIGHT))
    pygame.draw.line(screen, (90, 90, 100), (SIDEBAR_WIDTH, 0), (SIDEBAR_WIDTH, config.HEIGHT), 2)

    title = config.font.render("WIN RATE", True, (255, 255, 255))
    screen.blit(title, ((SIDEBAR_WIDTH - title.get_width()) // 2, 12))

    total = round_stats["chaser_wins"] + round_stats["runner_wins"]
    chaser_pct = round_stats["chaser_wins"] * 100.0 / total if total > 0 else 0.0
    runner_pct = round_stats["runner_wins"] * 100.0 / total if total > 0 else 0.0

    map_name = os.path.basename(map_paths[current_map_index])
    map_text = config.font.render(f"Map: {map_name}", True, (200, 200, 210))
    screen.blit(map_text, (14, 46))

    def stat_block(y, role, color, pct):
        pygame.draw.rect(screen, color, (14, y + 4, 12, 12))
        label = config.font.render(role, True, (255, 255, 255))
        screen.blit(label, (34, y))
        pct_label = config.font.render(f"{pct:.0f}%", True, (255, 255, 255))
        screen.blit(pct_label, (SIDEBAR_WIDTH - 14 - pct_label.get_width(), y))

        # bar background + filled portion
        bar_w = SIDEBAR_WIDTH - 28
        bar_h = 14
        pygame.draw.rect(screen, (70, 70, 80), (14, y + 26, bar_w, bar_h))
        fill_w = int(bar_w * pct / 100.0)
        if fill_w > 0:
            pygame.draw.rect(screen, color, (14, y + 26, fill_w, bar_h))

    stat_block(100, "Chaser", (255, 80, 80), chaser_pct)
    stat_block(160, "Runner", (80, 120, 255), runner_pct)


def draw_ray_lines(agent, opponent, clear_color, opponent_color):
    # Visualises every radar ray for one agent (used in duel mode only).
    ray_angles = [i * 45 for i in range(config.NUM_RAYS)] + config.VISION_CONE
    ax = int(agent.pos.x) + SIDEBAR_WIDTH
    ay = int(agent.pos.y)

    for angle in ray_angles:
        ray_dir = agent.dir.rotate(angle)
        endpoint = agent.pos + ray_dir * config.MAX_RADAR_DIST
        color = clear_color

        d = config.STEP_SIZE
        while d <= config.MAX_RADAR_DIST:
            ray_pos = agent.pos + ray_dir * d
            rx, ry = int(ray_pos.x), int(ray_pos.y)
            if 0 <= rx < config.WIDTH and 0 <= ry < config.HEIGHT:
                if map_mask.get_at((rx, ry)):
                    endpoint = ray_pos
                    color = (110, 110, 110)
                    break
            else:
                endpoint = ray_pos
                color = (110, 110, 110)
                break
            if ray_pos.distance_to(opponent.pos) < 10:
                endpoint = ray_pos
                color = opponent_color
                break
            d += config.STEP_SIZE

        pygame.draw.line(screen, color, (ax, ay), (int(endpoint.x) + SIDEBAR_WIDTH, int(endpoint.y)), 1)


# main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill('White')
    screen.blit(map_surface, (SIDEBAR_WIDTH, 0))

    ray_angles = [i * 45 for i in range(config.NUM_RAYS)] + config.VISION_CONE
    alive_runners = [r for r in runners if getattr(r, 'alive', True)]

    for i in range(population_size):
        chaser = chasers[i]
        runner = runners[i]

        if not getattr(runner, 'alive', True):
            continue

        # 1. Chaser raycasting & movement
        chaser_inputs = []
        for angle in ray_angles:
            ray_dir = chaser.dir.rotate(angle)
            ray_pos = chaser.pos.copy()
            current_dist = 0
            wall_dist = 1.0
            runner_dist = 1.0

            while current_dist < config.MAX_RADAR_DIST:
                current_dist += config.STEP_SIZE
                ray_pos = chaser.pos + (ray_dir * current_dist)
                rx, ry = int(ray_pos.x), int(ray_pos.y)

                if 0 <= rx < config.WIDTH and 0 <= ry < config.HEIGHT:
                    if map_mask.get_at((rx, ry)):
                        wall_dist = current_dist / config.MAX_RADAR_DIST
                        break
                else:
                    wall_dist = current_dist / config.MAX_RADAR_DIST
                    break

                if ray_pos.distance_to(runner.pos) < 10:
                    runner_dist = current_dist / config.MAX_RADAR_DIST
                    break

            chaser_inputs.extend([wall_dist, runner_dist])

        # Homing sense: relative direction to + distance from the runner.
        dx = (runner.pos.x - chaser.pos.x) / config.WIDTH
        dy = (runner.pos.y - chaser.pos.y) / config.HEIGHT
        dist_norm = min(1.0, chaser.pos.distance_to(runner.pos) / config.MAX_RADAR_DIST)
        chaser_inputs.extend([dx, dy, dist_norm])

        c_out_x, c_out_y = chaser.brain.forward(chaser_inputs)
        chaser.dir.rotate_ip(c_out_y * config.chaser_turn_speed)
        chaser_next = chaser.pos + chaser.dir * (c_out_x * config.chaser_speed)
        cnx, cny = int(chaser_next.x), int(chaser_next.y)

        # wall penalty for chaser
        if 0 <= cnx < config.WIDTH and 0 <= cny < config.HEIGHT and not map_mask.get_at((cnx, cny)):
            chaser.pos = chaser_next
        else:
            chaser.fitness -= 5.0

        chaser.pos.x = max(10, min(config.WIDTH - 10, chaser.pos.x))
        chaser.pos.y = max(10, min(config.HEIGHT - 10, chaser.pos.y))

        # 2. Runner raycasting & movement
        runner_inputs = []
        for angle in ray_angles:
            ray_dir = runner.dir.rotate(angle)
            ray_pos = runner.pos.copy()
            current_dist = 0
            wall_dist = 1.0
            chaser_dist = 1.0

            while current_dist < config.MAX_RADAR_DIST:
                current_dist += config.STEP_SIZE
                ray_pos = runner.pos + (ray_dir * current_dist)
                rx, ry = int(ray_pos.x), int(ray_pos.y)

                if 0 <= rx < config.WIDTH and 0 <= ry < config.HEIGHT:
                    if map_mask.get_at((rx, ry)):
                        wall_dist = current_dist / config.MAX_RADAR_DIST
                        break
                else:
                    wall_dist = current_dist / config.MAX_RADAR_DIST
                    break

                if ray_pos.distance_to(chaser.pos) < 10:
                    chaser_dist = current_dist / config.MAX_RADAR_DIST
                    break

            runner_inputs.extend([wall_dist, chaser_dist])

        r_out_x, r_out_y = runner.brain.forward(runner_inputs)
        runner.dir.rotate_ip(r_out_y * config.turn_speed)
        runner_next = runner.pos + runner.dir * (r_out_x * config.runner_speed)
        rnx, rny = int(runner_next.x), int(runner_next.y)

        # wall penalty for runner
        if 0 <= rnx < config.WIDTH and 0 <= rny < config.HEIGHT and not map_mask.get_at((rnx, rny)):
            runner.pos = runner_next
        else:
            runner.fitness -= 10.0

        runner.pos.x = max(10, min(config.WIDTH - 10, runner.pos.x))
        runner.pos.y = max(10, min(config.HEIGHT - 10, runner.pos.y))

        # fitness & tag collision
        runner.fitness += 1.0

        # Chaser reward is shaped by proximity: +10 when super close, -10 when too far.
        dist = chaser.pos.distance_to(runner.pos)
        close_dist = 50.0
        far_dist = float(config.MAX_RADAR_DIST)
        closeness = max(0.0, min(1.0, (far_dist - dist) / (far_dist - close_dist)))
        chaser.fitness += -10.0 + 20.0 * closeness

        # Bonus for actually closing the gap since the last tick (dense pursuit reward).
        if chaser.last_dist is not None:
            gap_closed = chaser.last_dist - dist
            if gap_closed > 0:
                chaser.fitness += config.CLOSE_PROGRESS_REWARD * min(gap_closed, 8.0)
        chaser.last_dist = dist

        if dist < 20:
            runner.alive = False
            seconds_left = round_timer - (pygame.time.get_ticks() - start_time) / 1000.0
            chaser.fitness += config.TAG_BASE_REWARD + config.TAG_EARLY_BONUS_PER_SEC * max(0.0, seconds_left)

        # draw rays in duel mode
        if not is_training_mode:
            draw_ray_lines(chaser, runner, (255, 180, 60), (0, 0, 255))
            draw_ray_lines(runner, chaser, (60, 220, 90), (255, 60, 60))

        # draw agents
        pygame.draw.circle(screen, (0, 0, 255), (int(runner.pos.x) + SIDEBAR_WIDTH, int(runner.pos.y)), 10)
        pygame.draw.circle(screen, (255, 0, 0), (int(chaser.pos.x) + SIDEBAR_WIDTH, int(chaser.pos.y)), 10)

    # round timer
    elapsed_seconds = (pygame.time.get_ticks() - start_time) / 1000
    time_remaining = max(0, int(round_timer - elapsed_seconds))

    if time_remaining <= 0 or len(alive_runners) == 0:
        record_round_results()
        if is_training_mode:
            evolve_population()
        else:
            reset_round()

    # HUD display
    if is_training_mode:
        hud_str = f"TRAINING MODE | Gen: {config.generation} | Alive: {len(alive_runners)}/{population_size} | Time: {time_remaining}s"
    else:
        hud_str = f"1v1 EVALUATION MODE | Trained Generations: {total_trained_generations} | Time: {time_remaining}s"

    hud_text = config.font.render(hud_str, True, (0, 0, 0))
    screen.blit(hud_text, (SIDEBAR_WIDTH + 10, 10))

    draw_sidebar()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()