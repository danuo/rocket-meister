import numpy as np
import pygame, os
from rocket_gym import RocketMeister10

# set the path of the states
# all states in the folder will be rendered
states_path = os.path.join('exported_states','')    

# each rocket will be color coded according to the replays filename (prefix)
# and the following dict: 
colordict = {
    'human': '#ffffff',
    'APPO': '#1a0e0e',
    'ARS': '#440a8e',
    'DDPG': '#ab0051', 
    'ES': '#ff5000',
    'MARWIL': '#3ad7c6',
    'PG': '#00949e',
    'PPO': '#8ee81f',
    'SAC': '#004d78',
    'TD3': '#650000',
}

filelist = [file for file in os.listdir(states_path) if file.endswith('.npy')]
print('loading these checkpoints:')
print(filelist)
filepathlist = [os.path.join(states_path,file) for file in filelist]
colorlist = []
for file in filelist:
    agent_name = file.split('_')[0]
    colorlist.append(colordict[agent_name])
numpy_import = np.load(filepathlist[0])   
statematrix = np.ones((numpy_import.shape[0], 7, len(filepathlist)))
for i, file in enumerate(filepathlist):
    numpy_import = np.load(file)
    statematrix[:,:,i] = numpy_import
    # change collided rockets to black color black (state = 3)
    statematrix[1:, -1, :] = np.where(statematrix[1:, 0, :] == 0, 3, statematrix[1:, -1, :])

# ─── RUN GAME FOR HUMAN MODE ────────────────────────────────────────────────────
env_config = {
    # "max_frames": 1000,
    # "export_frames": True,
    "export_states": False,
    "export_string": 'render',
    "gui_reward_total": False,
    }
    
env = RocketMeister10(env_config)
env.rule_collision = False
env.rule_timelimit = False
env.render()
run = True
counter = 0
while run:
    env.clock.tick(30)
    matrix = statematrix[counter,:,:].T
    if counter > 0:
        if np.count_nonzero(matrix[:,0].astype(int)) == 0:
            run = False
    env.set_spectator_state(matrix, colors=colorlist, frame=counter)
    env.render()
    counter += 1
input("Press Enter to exit...")