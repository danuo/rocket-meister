import gym
import numpy as np
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 1000


def line_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    # returns a (x, y) tuple or None if there is no intersection
    d = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if d:
        s = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / d
        t = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / d
    else:
        return None
    if not(0 <= s <= 1 and 0 <= t <= 1):
        return None
    x = x1 + s * (x2 - x1)
    y = y1 + s * (y2 - y1)
    return x, y

def line_intersect_front(x1, y1, x2, y2, x3, y3, x4, y4):
    d = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if d:
        s = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / d
        t = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / d
    else:
        return False
    if not(0 <= s and 0 <= t <= 1):
        return False
    x = x1 + s * (x2 - x1)
    y = y1 + s * (y2 - y1)
    return x, y


class Environment:
    def __init__(self, game):
        self.game = game
        # line is n*2 [x,y]
        self.L1_line1_array_source = np.array([[1195.,  986.], [1037.,  987.], [817.,  991.], [577.,  995.], [348.,  994.], [176.,  987.], [69.,  926.], [10.,  800.], [12.,  644.], [10.,  504.], [38.,  378.], [87.,  328.], [251.,  331.], [374.,  375.], [443.,  461.], [475.,  567.], [502.,  676.], [566.,  758.], [656.,  788.], [815.,  791.], [990.,  789.], [1112.,  757.], [1162.,  690.], [1168.,  615.], [1142.,  570.], [1098.,  544.], [1017.,  529.], [
                                           942.,  526.], [847.,  526.], [751.,  500.], [670.,  459.], [578.,  407.], [487.,  362.], [376.,  307.], [275.,  274.], [143.,  224.], [26.,   91.], [59.,   27.], [151.,    9.], [329.,    4.], [575.,    4.], [838.,    5.], [1018.,    3.], [1241.,    4.], [1375.,   18.], [1466.,   47.], [1525.,  129.], [1552.,  301.], [1578.,  463.], [1585.,  597.], [1591.,  747.], [1584.,  854.], [1529.,  917.], [1395.,  981.], [1195.,  986.]])
        self.L2_line1_array_source = np.array([[307,  972], [428,  974], [555,  967], [661,  961], [733,  941], [768,  905], [773,  840], [770,  768], [766,  703], [725,  629], [686,  569], [646,  530], [601,  496], [539,  464], [495,  422], [485,  387], [502,  362], [564,  350], [622,  369], [699,  414], [752,  477], [791,  566], [806,  656], [805,  732], [797,  820], [806,  918], [861,  976], [981,  985], [
                                           1101,  985], [1292,  983], [1497,  949], [1572,  804], [1560,  625], [1436,  547], [1279,  547], [1200,  540], [1179,  505], [1188,  470], [1214,  449], [1315,  443], [1391,  434], [1532,  332], [1564,  210], [1548,  102], [1366,   13], [1067,    9], [738,    5], [383,   12], [146,   21], [77,   42], [13,   98], [6,  224], [12,  411], [14,  586], [22,  806], [62,  897], [145,  949], [307,  972]])
        self.L1_line2_array_source = np.array([[200.,  685.], [209.,  659.], [218.,  652.], [240.,  649.], [269.,  661.], [291.,  703.], [323.,  758.], [392.,  832.], [446.,  865.], [576.,  881.], [775.,  881.], [958.,  879.], [1130.,  869.], [1243.,  831.], [1269.,  746.], [1290.,  652.], [1301.,  594.], [1300.,  518.], [1296.,  471.], [1282.,  421.], [1252.,  359.], [1215.,  321.], [1152.,  302.], [1087.,  290.], [983.,  286.], [
                                           867.,  282.], [769.,  266.], [657.,  246.], [591.,  232.], [553.,  205.], [561.,  173.], [582.,  161.], [689.,  144.], [800.,  136.], [953.,  123.], [1185.,  167.], [1249.,  253.], [1290.,  347.], [1311.,  447.], [1328.,  554.], [1330.,  664.], [1312.,  775.], [1265.,  855.], [1185.,  878.], [1082.,  894.], [785.,  894.], [443.,  892.], [349.,  855.], [258.,  786.], [210.,  744.], [200.,  708.], [200.,  685.]])
        self.L2_line2_array_source = np.array([[199,  425], [195,  565], [234,  700], [326,  767], [419,  760], [479,  708], [471,  607], [427,  530], [303,  392], [254,  297], [263,  233], [329,  180], [439,  163], [555,  166], [667,  176], [810,  227], [907,  305], [975,  422], [987,  478], [991,  582], [978,  677], [968,  771], [1026,  825], [1142,  829], [1302,  819], [
                                           1386,  784], [1368,  747], [1326,  723], [1199,  690], [1115,  676], [1036,  624], [1015,  546], [1018,  449], [1023,  339], [1095,  277], [1301,  254], [1401,  253], [1418,  235], [1404,  215], [1356,  185], [1270,  184], [1057,  198], [898,  189], [658,  140], [577,  134], [437,  134], [348,  137], [262,  169], [239,  208], [209,  332], [199,  425]])
        self.L1_goals_array_source = np.array([[626.95057785,    4.19753071,  636.16708048,  152.39401525], [772.05004586,    4.74923972,  769.5276892,  138.19620258], [941.04992788,    3.8550008,  924.71485104,  125.40331331], [1149.39178387,    3.58920082, 1084.44298632,  147.92884223], [1411.24197231,   29.54963953, 1211.72544361,  202.91231484], [1536.04778153,  199.37846013, 1267.69618596,  295.86442634], [1561.76535601,  361.84567977, 1306.04782318,  423.4182056], [1580.782591,  516.26674196, 1324.04197837,  529.08774624], [1587.37031445,  656.25786122, 1329.41387446,  631.76309528], [1588.8272306,  780.21233228, 1322.73241402,  708.8167802], [1445.20170707,  957.02306528, 1282.18682459,  825.74583049], [1173.24941961,  986.1376619, 1150.34743446,  883.3829228], [968.41227845,  988.24704948,  960.21693169,  894.], [763.36287101,  991.89395215,  760.47503118,  893.85657913], [579.15229047,  994.96412849,  582.08607121,  892.81336884], [379.11749456,  994.13588426,  418.22673035,  882.24881939], [128.12395417,  959.70617948,  290.37309477,  810.5466323], [10.25747935,  779.91661082,  204.7732576,  725.18372735], [11.87357409,  635.1501862,  203.4260307,  675.10257796], [28.57021131,  420.43404909,  232.17208098,  650.0674435], [
                                           253.56714949,  331.9183299,  250.37068754,  653.29131898], [435.03886981,  451.07743193,  281.72449704,  685.29222162], [484.02979196,  603.4536046,  299.45857884,  717.53818238], [515.79324148,  693.67259065,  353.33919842,  790.53769106], [558.1174322,  747.90046,  433.61740246,  857.43285706], [621.2977009,  776.43256697,  582.87037845,  881.], [766.477256,  790.08447653,  763.33259737,  881.], [901.66091439,  790.00958955,  907.68215498,  879.5499218], [1046.17270866,  774.26617478, 1090.09494996,  871.32006105], [1132.31357925,  729.7798038, 1249.71509973,  809.04678935], [1165.71998096,  643.50023798, 1290.05332246,  651.71884519], [1147.01950898,  578.68761169, 1299.58604403,  513.13601735], [1055.40942236,  536.11285599, 1199.31612608,  316.26994278], [890.16486682,  526.,  959.40019318,  285.18621356], [717.09005114,  482.8357049,  803.44855631,  271.62425409], [586.78822512,  411.96725768,  680.8765407,  250.26366798], [423.76586644,  330.66777166,  585.29922556,  227.94944974], [135.64401685,  215.63807043,  554.42991779,  199.28032885], [47.33717841,   49.61880551,  558.74480715,  182.02077139], [395.82028601,    4.,  571.68019127,  166.89703356]])
        self.load_level()

    def load_level(self):
        line1, line2, goals = np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,4))
        # default/level1 environment 
        if self.game.env_name in ['default', 'level1']:
            line1 = self.L1_line1_array_source.copy()
            line2 = self.L1_line2_array_source.copy()
            goals = self.L1_goals_array_source.copy()
            
        # level2 environment
        elif self.game.env_name == 'level2':
            line1 = self.L2_line1_array_source.copy()
            line2 = self.L2_line2_array_source.copy()
            # goals = self.L2_goals_array_source.copy()
            goals = np.zeros((2,4))
        
        # environment = empty: level with no boundaries
        elif self.game.env_name == 'empty':
            self.game.gui_draw_echo_points = False
            self.game.gui_draw_echo_vectors = False
            self.game.gui_draw_goal_all = False
            self.game.gui_draw_goal_next = False
            self.game.gui_draw_goal_points = False
        
        # environment = random: random level is generated
        elif self.game.env_name == 'random':
            # generate level and apply
            line1, line2, goals = self.generate_level_vectors_random(
                n_max = self.game.env_random_length)
            if self.game.camera_mode == 'fixed':
                print("When using env_name = 'random', the use of ",
                      "camera_mode = 'centered' is recommended.")
            
        if self.game.env_flipped:
            line1[:, 0] = -line1[:, 0] + WINDOW_WIDTH
            line2[:, 0] = -line2[:, 0] + WINDOW_WIDTH
            goals[:, [0,2]] = -goals[:, [0,2]] + WINDOW_WIDTH

        self.set_level_vectors(line1, line2, goals)
        self.n_goals = goals.shape[0]
        self.generate_collision_vectors(line1,line2)

    def move_env(self,d_x,d_y):
        # move the environment in fixed camera mode
        self.line1[:,0] = self.line1[:,0] - d_x
        self.line1[:,1] = self.line1[:,1] - d_y
        self.line2[:,0] = self.line2[:,0] - d_x
        self.line2[:,1] = self.line2[:,1] - d_y
        self.line1_list = self.line1.tolist()
        self.line2_list = self.line2.tolist()
        self.goals[:,[0,2]] = self.goals[:,[0,2]] - d_x
        self.goals[:,[1,3]] = self.goals[:,[1,3]] - d_y
        self.level_collision_vectors[:,[0,2]] = self.level_collision_vectors[:,[0,2]] - d_x
        self.level_collision_vectors[:,[1,3]] = self.level_collision_vectors[:,[1,3]] - d_y
    
    def set_level_vectors(self, line1, line2, goals, level_collision_vectors = None):
        self.line1 = line1
        self.line2 = line2
        self.goals = goals
        # list for pygame draw
        self.line1_list = line1.tolist()
        self.line2_list = line2.tolist()
        if level_collision_vectors:
            self.level_collision_vectors = level_collision_vectors

    def generate_collision_vectors(self,line1,line2):
        # for collision calculation, is numpy array
        # only call once to generate single line structe
        n1, n2 = line1.shape[0], line2.shape[0]
        line_combined = np.zeros((n1 + n2 - 2, 4))        
        line_combined[:n1-1,[0,1]] = line1[:n1-1,[0,1]]
        line_combined[:n1-1,[2,3]] = line1[1:n1,[0,1]]
        line_combined[n1-1:n1+n2-2,[0,1]] = line2[:n2-1,[0,1]]
        line_combined[n1-1:n1+n2-2,[2,3]] = line2[1:n2,[0,1]]
        self.level_collision_vectors = line_combined

    def get_goal_line(self, level):
        return self.goals[level, :]

    def generate_level_vectors_random(self, n_max = 50, steps_back=10):
        width = 100 # 10
        width_min = 40 # 10
        width_max = 150 # 25
        length_min = 60 # 30 # 5
        length_max = 150 # 20
        angle_mult = 0.5
        data = np.zeros((n_max, 7))
        data[0,0] = 50 # first x is shifted
        counter = 1  # dont change first
        while counter < n_max:
            sign = +1 if np.random.rand() > 0.5 else -1
            for point in range(np.random.randint(low=3, high=10)):
                ang_new = data[counter-1, 2] + sign * np.random.rand() * angle_mult
                if ang_new > np.pi:
                    ang_new -= 2 * np.pi
                if ang_new < -np.pi:
                    ang_new += 2 * np.pi
                data[counter, 2] = ang_new
                length = np.random.randint(low=length_min, high=length_max)
                x_old = data[counter-1, 0]
                y_old = data[counter-1, 1]
                x_new = x_old + length * np.cos(ang_new)
                y_new = y_old + length * np.sin(ang_new)
                data[counter, 0] = x_new
                data[counter, 1] = y_new
                for i in range(counter):
                    x3 = data[i, 0]
                    y3 = data[i, 0]
                    x4 = data[i+1, 0]
                    y4 = data[i+1, 0]
                    if line_intersect_front(x_old, y_old, x_new, y_new, x3, y3, x4, y4):
                        counter -= steps_back
                        counter = max(1,counter)
                        break
                # counter logic
                counter += 1
                if counter == n_max:
                    break
        # ─── CREATING LEFT AND RIGHT LINE ────────────────────────────────
        counter = 0
        while counter < n_max:
            sign = +1 if np.random.rand() > 0.4 else -1
            for point in range(np.random.randint(low=5, high=15)):
                width += sign * np.random.rand() * 5
                if width < width_min:
                    width = width_min
                    sign = +1
                if width > width_max:
                    width = width_max
                    sign = -1
                # width = max(5,min(20,width))
                data[counter, 3] = data[counter, 0] + width * np.cos(data[counter, 2]+1/2*np.pi)
                data[counter, 4] = data[counter, 1] + width * np.sin(data[counter, 2]+1/2*np.pi)
                data[counter, 5] = data[counter, 0] - width * np.cos(data[counter, 2]+1/2*np.pi)
                data[counter, 6] = data[counter, 1] - width * np.sin(data[counter, 2]+1/2*np.pi)
                counter += 1
                if counter == n_max:
                    break
        # todo: ─── IMPORTANT CALL AGAIN HERE if intersecting
        line1 = data[:,3:5]
        line1[:,0] = line1[:,0] + WINDOW_WIDTH//2
        line1[:,1] = line1[:,1] + WINDOW_HEIGHT//2
        line2 = data[:,5:7]
        line2[:,0] = line2[:,0] + WINDOW_WIDTH//2
        line2[:,1] = line2[:,1] + WINDOW_HEIGHT//2
        goals = np.zeros((n_max,4))
        
        goals[:,[0,1]] = line1.copy()
        goals[:,[2,3]] = line2.copy()
        
        add_l1 = np.array([[-100 + WINDOW_WIDTH//2,    0 + WINDOW_HEIGHT//2],
                           [ -50 + WINDOW_WIDTH//2,  100 + WINDOW_HEIGHT//2]])
        add_l2 = np.array([[-100 + WINDOW_WIDTH//2,    0 + WINDOW_HEIGHT//2],
                           [ -50 + WINDOW_WIDTH//2, -100 + WINDOW_HEIGHT//2]])
        add_g  = np.array([[-100 + WINDOW_WIDTH//2, -100 + WINDOW_HEIGHT//2,
                            -100 + WINDOW_WIDTH//2, +100 + WINDOW_HEIGHT//2],
                           [-.1  + WINDOW_WIDTH//2, -100 + WINDOW_HEIGHT//2,
                            -.11 + WINDOW_WIDTH//2, +100 + WINDOW_HEIGHT//2]])
        
        line1 = np.concatenate([add_l1,line1],axis=0)
        line2 = np.concatenate([add_l2,line2],axis=0)
        goals = np.concatenate([goals,add_g],axis=0)
        
        return line1, line2, goals


class Rocket:
    VEL_MAX = 15  # not used
    ROT_VEL = 0.08
    ACCELERATION = 0.5
    N_ECHO = 7  # must be odd

    def __init__(self, game, env):
        self.game = game
        self.env = env
        self.visible = True
        self.reset_game_state()

    def reset_game_state(self, x=500, y=100, ang=0, vel_x=0, vel_y=0, level=0):
        self.update_state(np.array([x, y, ang, vel_x, vel_y]))
        self.level = level
        self.level_previous = level
        # framecount_goal: since last goal
        self.framecount_goal = 0
        # framecount_total: since reset
        self.framecount_total = 0
        # reward: since last frame
        self.n_lap = 0
        self.reward_step = 0
        self.reward_total = 0
        self.done = False
        self.action = np.array([0, 0])
        self.action_state = 0
        self.update_echo_vectors()
        self.update_goal_vectors()
        self.check_collision_echo()

    def update_state(self, rocket_state):
        self.x = rocket_state[0]
        self.y = rocket_state[1]
        self.ang = rocket_state[2]
        self.vel_x = rocket_state[3]
        self.vel_y = rocket_state[4]

    def update_reward_continuous(self):
        reward_total_previous = self.reward_total
        if self.level == 0 and self.level_previous == self.env.n_goals-1:
            self.n_lap += 1
        if self.level == self.env.n_goals-1 and self.level_previous == 0:
            self.n_lap -= 1
        distance0 = np.sqrt((self.x-self.xi0)**2+(self.y-self.yi0)**2)
        distance1 = np.sqrt((self.x-self.xi1)**2+(self.y-self.yi1)**2)
        self.reward_total = self.n_lap * self.env.n_goals + self.level + 1 * (distance0/(distance0+distance1))
        self.reward_step = self.reward_total - reward_total_previous

    def update_reward_dynamic(self):  # dynamic
        if (self.level - self.level_previous == 1) or (self.level == 0 and self.level_previous == self.env.n_goals-1):
            self.reward_step = max(1, (500 - self.framecount_goal)) / 500
            self.reward_total += self.reward_step
            self.framecount_goal = 0
        if (self.level - self.level_previous == -1) or (self.level == self.env.n_goals-1 and self.level_previous == 0):
            self.reward_step = - 1
            self.reward_total += self.reward_step
            self.framecount_goal = 0

    def update_reward_static(self):  # static
        reward_total_previous = self.reward_total
        if self.level == 0 and self.level_previous == self.env.n_goals-1:
            self.n_lap += 1
        if self.level == self.env.n_goals-1 and self.level_previous == 0:
            self.n_lap -= 1
        self.reward_total = self.n_lap * self.env.n_goals + self.level
        self.reward_step = self.reward_total - reward_total_previous

    def update_goal_vectors(self):
        self.goal_vector = self.env.get_goal_line(self.level)
        self.goal_vector_next = self.env.get_goal_line(self.level)
        self.goal_vector_last = self.env.get_goal_line(self.level-1)

    def update_echo_vectors(self):
        n = self.N_ECHO
        if n % 2 == 0: n = max(n-1, 3)  # make sure that n>=3 and odd
        n_sideangles = int((n-1)/2)  # 7 -> 3
        matrix = np.zeros((n, 4))
        matrix[:, 0] = int(self.x)
        matrix[:, 1] = int(self.y)
        # straight angle
        matrix[n_sideangles, 2] = int(self.x + 1500 * np.cos(self.ang))
        matrix[n_sideangles, 3] = int(self.y - 1500 * np.sin(self.ang))
        # angles from 90 deg to 0
        # ignore first angle
        angles = np.linspace(0, np.pi/2, n_sideangles+1)
        for i in range(n_sideangles):
            # first side
            matrix[i, 2] = int(self.x + 1500 * np.cos(self.ang + angles[i+1]))  # x2
            matrix[i, 3] = int(self.y - 1500 * np.sin(self.ang + angles[i+1]))  # y2
            # second side
            matrix[-(i+1), 2] = int(self.x + 1500 * np.cos(self.ang - angles[i+1]))  # x2
            matrix[-(i+1), 3] = int(self.y - 1500 * np.sin(self.ang - angles[i+1]))  # y2
        self.echo_vectors = matrix

    def rotate(self, rotate):  # input: action1
        self.ang = self.ang + self.ROT_VEL * rotate
        # get angular in range of -pi,pi
        if self.ang > np.pi:
            self.ang = self.ang - 2 * np.pi
        if self.ang < -np.pi:
            self.ang = self.ang + 2 * np.pi

    def accelerate(self, accelerate):  # input: action0
        # backwards at half speed
        if accelerate < 0:
            accelerate = accelerate * 0.5

        # * velocity in abhängigkeit von ANG, also raketen stellung
        self.vel_x = self.vel_x + accelerate * np.cos(self.ang)
        self.vel_y = self.vel_y - accelerate * np.sin(self.ang)

        # * cap auf max speed MAX_VEL
        # if np.sqrt(self.vel_x**2 + self.vel_y**2) > self.MAX_VEL:
        #     # * verteilung der self.MAX_VEL anteilig
        #     self.vel_x = self.MAX_VEL * self.vel_x / \
        #         (np.abs(self.vel_x)+np.abs(self.vel_y))
        #     self.vel_y = self.MAX_VEL * self.vel_y / \
        #         (np.abs(self.vel_x)+np.abs(self.vel_y))

    def update_observations(self):
        # ─── OBSERVATION 8: VELOCITY ─────────────────────────────────────
        vel = np.sqrt(self.vel_x**2 + self.vel_y**2)
        self.vel_interp = np.interp(vel, [0, 50], [-1, 1])

        # ─── OBSERVATION 9: VELOCITY ANGLE ───────────────────────────────
        # get angular difference
        vel_ang = np.arctan2(-self.vel_y, self.vel_x)
        vel_ang_diff = self.ang - vel_ang

        # set between -pi and pi
        if vel_ang_diff > np.pi:
            vel_ang_diff = vel_ang_diff - 2 * np.pi
        if vel_ang_diff < -np.pi:
            vel_ang_diff = vel_ang_diff + 2 * np.pi
        if self.vel_interp < 0.001 - 1:
            vel_ang_diff = 0

        # normalize
        self.vel_ang_diff_interp = np.interp(vel_ang_diff, [-np.pi, np.pi], [-1, 1])

        # ─── OBSERVATION 10: GOAL ANGLE ──────────────────────────────────
        def get_intersection_point(xp, yp, x1, y1, x2, y2):
            # check if line is vertical (infinite slope)
            if x1 == x2:
                return(x1,yp)
            # slope: dy/dx
            a = (y2-y1) / (x2-x1)
            b = y1 - a * x1
            xi = (xp * (1/a) + yp - b) * 1/(a + 1/a)
            yi = a * xi + b
            return xi, yi

        # from rocket to next goal line direction
        goal_next = self.goal_vector_next
        goal_last = self.goal_vector_last

        xp, yp = self.x, self.y
        
        x1, y1, x2, y2 = goal_last
        self.xi0, self.yi0 = get_intersection_point(xp, yp, x1, y1, x2, y2)
        
        x1, y1, x2, y2 = goal_next
        self.xi1, self.yi1 = get_intersection_point(xp, yp, x1, y1, x2, y2)

        dx, dy = self.xi1 - self.x, self.yi1 - self.y
        goal_ang = np.arctan2(-dy, dx)
        goal_ang_diff = self.ang - goal_ang

        if goal_ang_diff > np.pi:
            goal_ang_diff = goal_ang_diff - 2 * np.pi
        if goal_ang_diff < -np.pi:
            goal_ang_diff = goal_ang_diff + 2 * np.pi

        self.goal_ang_diff_interp = np.interp(goal_ang_diff, [-np.pi, np.pi], [-1, 1])

    def move(self, action):
        # first apply rotation!
        self.rotate(action[1])
        self.accelerate(action[0])

        # displacement
        d_x, d_y = self.vel_x, self.vel_y
        x_from, y_from = self.x, self.y

        # ─── CENTERED MODE ───────────────────────────────────────────────
        if self.game.camera_mode == 'centered':
            self.env.move_env(d_x,d_y)
            self.movement_vector = [WINDOW_WIDTH/2, WINDOW_HEIGHT/2, WINDOW_WIDTH/2 + d_x, WINDOW_HEIGHT/2 + d_y]
        # ─── FIXED MODE ─────────────────────────────────────────────────
        if self.game.camera_mode == 'fixed':
            self.x = self.x + d_x
            self.y = self.y + d_y
            self.movement_vector = [x_from, y_from, self.x, self.y]

        # ─── KEEP ON SCREEN ──────────────────────────────────────────────
        # rocket cannot leave fixed screen area
        if self.game.rule_keep_on_screen:
            if self.x > WINDOW_WIDTH:
                self.x = self.x - WINDOW_WIDTH
            elif self.x < 0:
                self.x = self.x + WINDOW_WIDTH
            if self.y > WINDOW_HEIGHT:
                self.y = self.y - WINDOW_HEIGHT
            elif self.y < 0:
                self.y = self.y + WINDOW_HEIGHT

    def check_collision_goal(self):
        result_last = line_intersect(*self.movement_vector, *self.goal_vector_last)
        result_next = line_intersect(*self.movement_vector, *self.goal_vector_next)
        if result_last is not None:
            self.level -= 1
            if self.level == -1:
                self.level = self.env.n_goals - 1
            self.update_goal_vectors()
        elif result_next is not None:
            self.level += 1
            if self.level == self.env.n_goals:
                self.level = 0
            if self.game.env == 'random':
                if self.level == self.game.env_random_length:
                    self.game.set_done()
            self.update_goal_vectors()

    def check_collision_env(self):
        for line in self.env.level_collision_vectors:
            result = line_intersect(*self.movement_vector, *line)
            if result is not None:
                self.game.set_done()
                break

    def check_collision_echo(self):
        # max_distance: Distance value maps to observation=1 if distance >= max_distance
        max_distance = 5000 
        points = np.full((self.N_ECHO, 2), self.x) # points for visualiziation
        points[:,1] = self.y
        distances = np.full((self.N_ECHO), max_distance) # distances for observation
        n = self.env.level_collision_vectors.shape[0]
        for i in range(self.N_ECHO):
            found = False
            line1 = self.echo_vectors[i, :]
            points_candidates = np.zeros((n,2))
            distances_candidates = np.full((n), max_distance)
            for j, line2 in enumerate(self.env.level_collision_vectors):
                result = line_intersect(*line1, *line2)
                if result is not None:
                    found = True
                    points_candidates[j,:] = result
                    distances_candidates[j] = np.sqrt((self.x-result[0])**2+(self.y-result[1])**2)
            if found: # make sure one intersection is found
                argmin = np.argmin(distances_candidates)  # index of closest intersection 
                points[i, :] = points_candidates[argmin]
                distances[i] = distances_candidates[argmin]

        self.echo_collision_points = points
        # ─── NORMALIZE DISTANCES ─────────────────────────────────────────
        # linear mapping from 0,1000 to -1,1
        # distance 0 becomes -1, distance 1000 becomes +1
        # values always in range [-1,1]
        self.echo_collision_distances_interp = np.interp(distances, [0, 1000], [-1, 1])


#
# ─────────────────────────────────────────────────────────────────────────
#   :::::: R O C K E T G A M E 10 : :  :   :    :     :        :          :
# ─────────────────────────────────────────────────────────────────────────
#
class RocketMeister10(gym.Env):
    def __init__(self, env_config={}):
        self.parse_env_config(env_config)
        self.win = None
        self.action_space = gym.spaces.Box(
            low=-1.,
            high=1.,
            shape=(2,),
            dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-1.,
            high=1.,
            shape=(10,),
            dtype=np.float32)

        self.env = Environment(self)
        self.rocket = Rocket(self, self.env)
        self.spectator = None

        self.reset()
        # exit()

    def parse_env_config(self,env_config):
        keyword_dict = {
            # these are all available keyboards and valid values respectively
            # the first value in the list is the default value
            'gui'                   : [True, False],
            'camera_mode'           : ['fixed','centered'],
            'env_name'              : ['default', 'empty', 'level1', 'level2', 'random'],
            'env_random_length'     : [50, 'any', int],                    # length of randomly generated environment
            'env_flipped'           : [False, True],                       # activates normal environment, flipped
            'env_flipmode'          : [False, True],                       # activates flip mode. Each reset() flips env
            'env_visible'           : [True, False],                 
            'reward_mode'           : ['continuous', 'static', 'dynamic'], # choose reward mode
            'export_frames'         : [False, True],                       # export rendered frames
            'export_states'         : [False, True],                       # export every step
            'export_string'         : ['', 'any', str],                    # string for export filename
            'export_highscore'      : [0, 'any', int],                     # only export if highscore is beat
            'max_steps'             : [1000, 'any', int],
            'rule_collision'        : [True, False],
            'rule_max_steps'        : [True, False],
            'rule_keep_on_screen'   : [False, True],
            'gui_echo_distances'    : [False, True],
            'gui_frames_remaining'  : [True, False],
            'gui_goal_ang'          : [False, True],
            'gui_level'             : [False, True],
            'gui_reward_total'      : [True, False],
            'gui_velocity'          : [False, True],
            'gui_draw_echo_points'  : [True, False],
            'gui_draw_echo_vectors' : [False, True],
            'gui_draw_goal_all'     : [True, False],
            'gui_draw_goal_next'    : [True, False],
            'gui_draw_goal_points'  : [False, True],
        }
        
        # ─── STEP 1 GET DEFAULT VALUE ────────────────────────────────────
        assign_dict = {}
        for keyword in keyword_dict:
            # asign default value form keyword_dict
            assign_dict[keyword] = keyword_dict[keyword][0]
            
        # ─── STEP 2 GET VALUE FROM env_config ─────────────────────────────
        for keyword in env_config:
            if keyword in keyword_dict:
                # possible keyword proceed with assigning
                if env_config[keyword] in keyword_dict[keyword]:
                    # valid value passed, assign
                    assign_dict[keyword] = env_config[keyword]
                elif 'any' in keyword_dict[keyword]:
                    # any value is allowed, assign if type matches
                    if isinstance(env_config[keyword],keyword_dict[keyword][2]):
                        print('type matches')
                        assign_dict[keyword] = env_config[keyword]
                    else:
                        print('error: wrong type. type needs to be: ', keyword_dict[keyword][2])
                else:
                    print('given keyword exists, but given value is illegal')
            else:
                print('passed keyword does not exist: ',keyword)

        # ─── ASSIGN DEFAULT VALUES ───────────────────────────────────────
        self.camera_mode           = assign_dict['camera_mode']
        self.env_name              = assign_dict['env_name']
        self.env_random_length     = assign_dict['env_random_length']
        self.env_flipped           = assign_dict['env_flipped']
        self.env_flipmode          = assign_dict['env_flipmode']
        self.env_visible           = assign_dict['env_visible']
        self.reward_mode           = assign_dict['reward_mode']
        self.export_frames         = assign_dict['export_frames']
        self.export_states         = assign_dict['export_states']
        self.export_string         = assign_dict['export_string']
        self.export_highscore      = assign_dict['export_highscore']
        self.max_steps             = assign_dict['max_steps']
        self.rule_collision        = assign_dict['rule_collision']
        self.rule_max_steps        = assign_dict['rule_max_steps']
        self.rule_keep_on_screen   = assign_dict['rule_keep_on_screen']
        self.gui                   = assign_dict['gui']
        self.gui_echo_distances    = assign_dict['gui_echo_distances']
        self.gui_frames_remaining  = assign_dict['gui_frames_remaining']
        self.gui_goal_ang          = assign_dict['gui_goal_ang']
        self.gui_level             = assign_dict['gui_level']
        self.gui_reward_total      = assign_dict['gui_reward_total']
        self.gui_velocity          = assign_dict['gui_velocity']
        self.gui_draw_echo_points  = assign_dict['gui_draw_echo_points']
        self.gui_draw_echo_vectors = assign_dict['gui_draw_echo_vectors']
        self.gui_draw_goal_all     = assign_dict['gui_draw_goal_all']
        self.gui_draw_goal_next    = assign_dict['gui_draw_goal_next']
        self.gui_draw_goal_points  = assign_dict['gui_draw_goal_points']
        

    def reset(self):
        # ─── FLIP MIRROR ─────────────────────────────────────────────────
        if self.env_flipmode:
            if self.env_flipped:
                self.env_flipped = False
            else:
                self.env_flipped = True
            self.env.load_level()
            
        # if self.env == 'random' or self.camera_mode == 'centered':
        if self.camera_mode == 'centered':
            self.env.load_level()

        # ─── RESET EXPORT VARIALBES ──────────────────────────────────────
        # give unique session id for export
        self.session_id = str(int(np.random.rand(1)*10**6)).zfill(6)
        # dim0 : n_steps | dim1 : frame, x,y,ang,velx,vely
        self.statematrix = np.zeros((self.max_steps, 7))

        # ─── RESET rocket ──────────────────────────────────────────────────
        self.reset_rocket_state()
        # generate observation
        self.rocket.update_observations()
        distances = self.rocket.echo_collision_distances_interp
        velocity = self.rocket.vel_interp
        vel_ang_diff = self.rocket.vel_ang_diff_interp
        goal_ang_diff = self.rocket.goal_ang_diff_interp
        observation10 = np.concatenate((distances, np.array([velocity, vel_ang_diff, goal_ang_diff])))
        return observation10

    def set_spectator_state(self, state, colors=[], frame=None):
        self.rocket.visible = False
        self.spectator = state
        self.spectator_colorlist = colors
        if frame:
            self.rocket.framecount_total = frame

    def reset_rocket_state(self, x=500, y=100, ang=1e-9, vel_x=0, vel_y=0, level=0):  # ang=1e-10
        if self.env == 'random':
            x, y = WINDOW_WIDTH//2, WINDOW_HEIGHT//2
        elif self.env_flipped:
            # mirror over y-axis
            x, ang, vel_x = -x+WINDOW_WIDTH, np.pi-ang, -vel_x
        # if camera_mode is centerd, the rocket needs to go center too
        if self.camera_mode == 'centered':
            diff_x = WINDOW_WIDTH//2  - x
            diff_y = WINDOW_HEIGHT//2 - y
            # move environment
            if self.env_name in ['default', 'level1', 'level2']:
                self.env.move_env(-diff_x,-diff_y)
            # move player
            x, y = WINDOW_WIDTH//2, WINDOW_HEIGHT//2
        return self.rocket.reset_game_state(x, y, ang, vel_x, vel_y, level)

    def set_done(self):
        self.rocket.done = True
        if (self.export_states) and (self.rocket.reward_total > self.export_highscore):
            import os
            # copy last state to remaining frames
            i = self.rocket.framecount_total
            n_new = self.max_steps - i
            self.statematrix[i:, :] = np.repeat(self.statematrix[i-1, :].reshape((1, 7)), n_new, axis=0)
            # mark at which frame agent is done
            self.statematrix[i:, 0] = 0

            # export
            filename = '_'.join([self.export_string,
                                 '-'.join([self.session_id, str(int(self.rocket.reward_total)).zfill(4)])
                                 ])
            filenamepath = os.path.join('exported_states', filename)
            np.save(filenamepath, self.statematrix)

    def step(self, action=[0, 0]):
        # ─── NORMALIZE ACTION ────────────────────────────────────────────
        # action = [action_acc, action_turn]
        action[0] = max(min(action[0], 1), -1)
        action[1] = max(min(action[1], 1), -1)

        # set action_state for image
        self.rocket.action = action.copy()
        if self.rocket.action[0] < 0:
            self.rocket.action_state = 2
        else:
            self.rocket.action_state = 0
        if self.rocket.action[0] > 0:
            self.rocket.action_state = 1

        # ─── PERFORM STEP ────────────────────────────────────────────────
        if not self.rocket.done:
            self.rocket.move(action)
            self.rocket.update_echo_vectors()
            if self.rule_collision:
                self.rocket.check_collision_goal()
                self.rocket.check_collision_echo()
                self.rocket.check_collision_env()
            self.rocket.update_observations()

            # ─── EXPORT GAME STATE ───────────────────────────────────────────
            if self.export_states:
                i = self.rocket.framecount_total
                # frame, x,y,ang,velx,vely
                self.statematrix[i, :] = [i, self.rocket.x, self.rocket.y, self.rocket.ang,
                                             self.rocket.vel_x, self.rocket.vel_y, self.rocket.action_state]

            self.rocket.framecount_goal += 1
            self.rocket.framecount_total += 1

            if self.reward_mode == 'static':
                self.rocket.update_reward_static()
            elif self.reward_mode == 'dynamic':
                self.rocket.update_reward_dynamic()
            else:
                # make default
                self.rocket.update_reward_continuous()

            if self.rule_max_steps:
                if self.rocket.framecount_total == self.max_steps - 1:
                    self.set_done()

        # ─── GET RETURN VARIABLES ────────────────────────────────────────
        distances = self.rocket.echo_collision_distances_interp
        velocity = self.rocket.vel_interp
        vel_ang_diff = self.rocket.vel_ang_diff_interp
        goal_ang_diff = self.rocket.goal_ang_diff_interp

        observation10 = np.concatenate((distances, np.array([velocity, vel_ang_diff, goal_ang_diff])))

        reward = self.rocket.reward_step
        done = self.rocket.done
        info = {
            "x": self.rocket.x,
            "y": self.rocket.y,
            "ang": self.rocket.ang}

        # ─── RESET ITERATION VARIABLES ───────────────────────────────────
        self.rocket.reward_step = 0
        self.rocket.level_previous = self.rocket.level
        return observation10, reward, done, info

    def render(self, mode=None):
        # initialize pygame only when render is called once
        import pygame
        import os
        from PIL import Image
        middle_echo_index = (self.rocket.N_ECHO - 1) // 2

        def init_renderer(self):
            self.ROCKET_IMG = [pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'rocket_no_power.png'))), pygame.transform.scale2x(pygame.image.load(os.path.join(
                'imgs', 'rocket_power.png'))), pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'rocket_power_front.png'))), pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'rocket_black.png')))]
            self.BG_IMG = pygame.image.load(os.path.join('imgs', 'space_wp3.jpg'))
            pygame.display.set_caption("Flappy Rocket")
            self.clock = pygame.time.Clock()
            self.win = pygame.display.set_mode(
                (WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.init()
            
            if self.export_frames:
                self.display_surface = pygame.display.get_surface()
                self.image3d = np.ndarray(
                    (WINDOW_WIDTH, WINDOW_HEIGHT, 3), np.uint8)
                
            self.gui_interface = []
            if self.gui_reward_total:
                self.gui_interface.append('reward_total')
            if self.gui_level:
                self.gui_interface.append('level')
            if self.gui_echo_distances:
                self.gui_interface.append('echo_distances')
            if self.gui_goal_ang:
                self.gui_interface.append('goal_ang')
            if self.gui_velocity:
                self.gui_interface.append('velocity')
            if self.gui_frames_remaining:
                self.gui_interface.append('frames_remaining')
            
            
            middle_echo_index = int((self.rocket.N_ECHO-1)/2)

        def draw_level():
            pygame.draw.lines(self.win, (175, 65, 255), False, self.env.line1_list, 4)
            pygame.draw.lines(self.win, (175, 65, 255), False, self.env.line2_list, 4)

        def draw_goal_next():
            goal = tuple(self.env.goals[self.rocket.level, :])
            # pygame.draw.lines(self.win, (60, 230, 255), False,
                            #   (goal[0:2],goal[2:4]), 4)
            pygame.draw.lines(self.win, (130, 210, 255), False,
                              (goal[0:2],goal[2:4]), 4)

        def draw_goal_all():
            for i in range(self.env.goals.shape[0]):
                goal = tuple(self.env.goals[i, :])
                pygame.draw.lines(self.win, (60, 180, 250), False,
                                  (goal[0:2],goal[2:4]),4)

        def draw_rocket():
            self.rocket.img = self.ROCKET_IMG[self.rocket.action_state]
            # pygame.transform.rotate takes angle in degree
            rotated_image = pygame.transform.rotate(
                self.rocket.img, self.rocket.ang / np.pi * 180)
            new_rect = rotated_image.get_rect(center=self.rocket.img.get_rect(
                center=(self.rocket.x, self.rocket.y)).center)
            self.win.blit(rotated_image, new_rect.topleft)

        def draw_spectators():
            if not self.rocket.visible and self.spectator is not None:
                for i, row in enumerate(self.spectator):
                    framecount_total, x, y, ang, vel_x, vel_y, action_state = row
                    image = self.ROCKET_IMG[int(action_state)]
                    # pygame.transform.rotate takes angle in degree
                    rotated_image = pygame.transform.rotate(
                        image, ang / np.pi * 180)
                    new_rect = rotated_image.get_rect(center=image.get_rect(
                        center=(x, y)).center)
                    self.win.blit(rotated_image, new_rect.topleft)
                    # color marker
                    if self.spectator_colorlist:
                        color = self.spectator_colorlist[i]
                        pygame.draw.circle(self.win, color, (int(x), int(y)), 10)

        def draw_goal_intersection_points():
            pygame.draw.circle(self.win, (250, 0, 250), (int(self.rocket.xi0), int(self.rocket.yi0)), 6)
            pygame.draw.circle(self.win, (250, 0, 250), (int(self.rocket.xi1), int(self.rocket.yi1)), 6)

        def draw_echo_vector():
            n =self.rocket.N_ECHO
            echo_vectors_short = self.rocket.echo_vectors
            if len(self.rocket.echo_collision_points) == n:
                echo_vectors_short = self.rocket.echo_vectors
                for i in range(n):
                    echo_vectors_short[i,[2,3]] = self.rocket.echo_collision_points[i]                    
            for vector in echo_vectors_short:
                pygame.draw.line(self .win, (135, 40, 160), vector[0:2], vector[2:4], 4)

        def draw_echo_collision_points():
            for point in self.rocket.echo_collision_points:
                pygame.draw.circle(self.win, (255, 40, 40), (int(point[0]), int(point[1])), 6)

        def draw_text(surface, text=None, size=30, x=0, y=0, 
                      font_name=pygame.font.match_font('consolas'), 
                      position='topleft'):
            font = pygame.font.Font(font_name, size)
            text_surface = font.render(text, True, (255, 255, 255))
            text_rect = text_surface.get_rect()
            if position == 'topleft':
                text_rect.topleft = (x, y)
            if position == 'topright':
                text_rect.topright = (x, y)
            surface.blit(text_surface, text_rect)
            
        def get_gui_value(value: str):
            if value == 'reward_total':
                return str(round(self.rocket.reward_total, 2))
            elif value == 'level':
                return str(self.rocket.level)
            elif value == 'echo_distances':
                return str(round(self.rocket.echo_collision_distances_interp[middle_echo_index], 2))
            elif value == 'velocity':
                return str(round(np.sqrt(self.rocket.vel_x**2+self.rocket.vel_y**2), 2))
            elif value == 'goal_ang':
                return str(round(self.rocket.goal_ang_diff_interp, 2))
            elif value == 'frames_remaining':
                return str(self.max_steps-self.rocket.framecount_total)
                # return str(self.rocket.framecount_total)
            else:
                return 'value not found'


        # ─── INIT RENDERER ───────────────────────────────────────────────
        if self.win is None:
            init_renderer(self)

        # ─── RECURING RENDERING ──────────────────────────────────────────
        self.win.blit(self.BG_IMG, (0, 0))
        if self.gui_draw_goal_all:
            draw_goal_all()
        if self.gui_draw_goal_next:
            draw_goal_next()
        if self.gui_draw_echo_points:
            draw_echo_collision_points()
        if self.gui_draw_echo_vectors:
            draw_echo_vector()
        if self.gui_draw_goal_points:
            draw_goal_intersection_points()
        if self.env_visible:
            draw_level()
        if self.rocket.visible:
            draw_rocket()
        draw_spectators()

        # ─── INTERFACE ───────────────────────────────────────────────────
        if self.gui:
            gui_n = len(self.gui_interface)
            gui_x_pad = 10
            if gui_n == 1:
                gui_x_list = [WINDOW_WIDTH-gui_x_pad]
            else:
                gui_x_list = np.linspace(0+gui_x_pad, WINDOW_WIDTH-gui_x_pad, gui_n)
            for i in range(gui_n):
                key = self.gui_interface[i]
                pos = 'topright' if (i==gui_n-1) else 'topleft'
                # draw key
                draw_text(self.win, text=key,
                    size=15, x=gui_x_list[i], y=8, position=pos)
                # draw value
                draw_text(self.win, text=get_gui_value(key),
                    size=30, x=gui_x_list[i], y=20, position=pos)

        # ─── RENDER GAME ─────────────────────────────────────────────────
        pygame.display.update()

        # ─── EXPORT GAME FRAMES ──────────────────────────────────────────
        if self.export_frames:
            pygame.pixelcopy.surface_to_array(
                self.image3d, self.display_surface)
            self.image3dT = np.transpose(self.image3d, axes=[1, 0, 2])
            im = Image.fromarray(self.image3dT)  # monochromatic image
            imrgb = im.convert('RGB')  # color image

            filename = ''.join([
                self.export_string,
                self.session_id,
                '-frame-',
                str(self.rocket.framecount_total).zfill(5),
                '.jpg'])
            filenamepath = os.path.join('exported_frames', filename)
            imrgb.save(filenamepath)

    def get_rocket_state(self):
        return np.array([
            self.rocket.x,
            self.rocket.y,
            self.rocket.ang,
            self.rocket.vel_x,
            self.rocket.vel_y,
        ])

    def update_rocket_state(self, rocket_state):
        self.rocket.update_state(rocket_state)

    def update_interface_vars(self, action_next):
        self.action_next = action_next


# ─── RocketMeister9 ────────────────────────────────────────────────────────────────
# 7distances, velocity, angdiff
class RocketMeister9(RocketMeister10):
    # * done
    def __init__(self, config={}):
        super().__init__(config)
        self.observation_space = gym.spaces.Box(
            low=-1.,
            high=1.,
            shape=(9,),
            dtype=np.float32)

    def reset(self):
        observation10 = super().reset()
        observation9 = observation10[:9]
        return observation9

    def step(self, action=[0, 0]):
        observation10, reward, done, info = super().step(action)
        observation9 = observation10[:9]
        return observation9, reward, done, info

# ─── RocketMeister8 ────────────────────────────────────────────────────────────────
# 7distances, velocity
class RocketMeister8(RocketMeister10):
    # * done
    def __init__(self, config={}):
        super().__init__(config)
        self.observation_space = gym.spaces.Box(
            low=-1.,
            high=1.,
            shape=(8,),
            dtype=np.float32)

    def reset(self):
        observation10 = super().reset()
        observation8 = observation10[:8]
        return observation8

    def step(self, action=[0, 0]):
        observation10, reward, done, info = super().step(action)
        observation8 = observation10[:8]
        return observation8, reward, done, info

# ─── RocketMeister7 ────────────────────────────────────────────────────────────────
# 7distances
class RocketMeister7(RocketMeister10):
    # * done
    def __init__(self, config={}):
        super().__init__(config)
        self.observation_space = gym.spaces.Box(
            low=-1.,
            high=1.,
            shape=(7,),
            dtype=np.float32)

    def reset(self):
        observation10 = super().reset()
        observation7 = observation10[:7]
        return observation7

    def step(self, action=[0, 0]):
        observation10, reward, done, info = super().step(action)
        observation7 = observation10[:7]
        return observation7, reward, done, info
