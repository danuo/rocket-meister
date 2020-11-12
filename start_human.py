import numpy as np
import pygame
from rocket_gym import RocketMeister10

# ─── FUNCTIONS FOR USER INPUT ───────────────────────────────────────────────────
def event_to_action(eventlist):
    global run
    for event in eventlist:
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            env.reset()

def pressed_to_action(keytouple):
    action_turn = 0.
    action_acc = 0.
    if keytouple[81] == 1:  # back
        action_acc -= 1
    if keytouple[82] == 1:  # forward
        action_acc += 1
    if keytouple[80] == 1:  # left  is -1
        action_turn += 1
    if keytouple[79] == 1:  # right is +1
        action_turn -= 1
    if keytouple[21] == 1:  # r, reset
        # game.reset()
        pass
    # ─── KEY IDS ─────────
    # arrow backwards : 81
    # arrow forward   : 82
    # arrow left      : 80
    # arrow right     : 79
    # r               : 21
    return np.array([action_acc, action_turn])

# ─── INITIALIZE AND RUN ENVIRONMENT ─────────────────────────────────────────────
env_config = {
    'gui': True,
    'env_name': 'default',
    # 'env_name': 'empty',
    # 'env_name': 'level1',
    # 'env_name': 'level2',
    # 'env_name': 'random',
    # 'camera_mode': 'centered',
    # 'env_flipped': False,
    # 'env_flipmode': False,
    # 'export_frames': True,
    'export_states': True,
    # 'export_highscore': False,
    'export_string': 'human',
    'max_steps': 1000,
    'gui_reward_total': True,
    'gui_echo_distances': True,
    'gui_velocity': True,
    'gui_goal_ang': True,
    'gui_frames_remaining': True,
    'gui_draw_echo_points': True,
    'gui_draw_echo_vectors': True,
    'gui_draw_goal_points': True,
}

env = RocketMeister10(env_config)
env.render()
run = True
while run:
    env.clock.tick(30)
    get_event = pygame.event.get()
    event_to_action(get_event)
    get_pressed = list(pygame.key.get_pressed())
    action = pressed_to_action(get_pressed)
    env.step(action=action)
    env.render()
pygame.quit()
