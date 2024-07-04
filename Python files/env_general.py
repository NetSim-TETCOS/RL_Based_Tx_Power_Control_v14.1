import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
import numpy as np
import torch
import random
import struct
import socket

register(
    id="general-v1",
    entry_point="env_general:generalized" #filename:classname
)

class generalized(gym.Env):
    def __init__(self,num_gNBs=3,num_UEs=6):
        super(generalized, self).__init__()
        
        # State parameters
        self.it=0
        self.num_gNBs = num_gNBs
        self.num_UEs = num_UEs
        self.adjustments = [-3, -1, 0, 1, 3]
        # self.adjustments = [0, 0, 0, 0, 0]
        self.num_actions=len(self.adjustments) ** self.num_gNBs
        self.current_gNB_powers = [40 for _ in range(self.num_gNBs)]
        self.total_rewards=[]
        # self.total_rewards=0

        # Define action space
        self.action_space = spaces.Discrete(self.num_actions)
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=-500.00,
            high=500.00,
            shape=(self.num_UEs,),
            dtype=np.float64,
        )
        # Interfacing variables
        self.port = 12346
        self.server_address = (socket.gethostbyname(socket.gethostname()), self.port)
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.client_socket.connect(self.server_address)
        self.bytes_to_receive = (2*self.num_UEs+1)*8
        self.seed_val = 42
        self._seed(self.seed_val)

        # Initial state
        self.state = np.zeros(self.num_UEs)

        # All constants for calculations
        # self.thermal_noise = -93.8568
        # self.TxPower = 40
        # self.freq_GHz = 3.5
        # self.freq_Hz = self.freq_GHz * 1e9
        # self.BW_MHz = 50
        # self.B_Hz = self.B_MHz * 1e6
        # self.eta_service = 2.0
        # self.d0 = 1
        # self.packet_size_bytes = 1500
        # self.c = 3*(1e8)
        # self.wavelength = self.c / self.freq_Hz
        # self.const_term = 20*np.log10((4*np.pi*self.d0)/self.wavelength)
        # self.path_losses_matrix = np.array([[self.log_pathloss(d) for d in row] for row in self.distances])
        # self.spectral_eff_list = [0.1523, 0.3770, 0.8770, 1.4766, 1.9141, 2.4063, 2.7305, 3.3223, 3.9023, 4.5234, 5.1152, 5.5547, 6.2266, 6.9141, 7.4063]
        # self.dl_ul_ratio = 4/5
        # self.overhead_multiplier = 0.86
        # self.app_by_phy = 0.70
        # self.ue_divider = 0.5
        # Store render_mode
        self.render_mode = None

    def _seed(self, seed=None):
        self.seed_val = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        self._seed(seed if seed is not None else self.seed_val)

        # Reset state
        self.current_gNB_powers = [40 for _ in range(self.num_gNBs)]
        self.total_rewards.clear()
        # self.total_rewards=0
        self.it=0
        # self.sinr_reset()
        # if options is None:
            
        # else:
        #     pass
        self.episode_Start()
        return self.state, {}
    
    def episode_Start(self):
        # self.client_socket.close()
        while True:
            try:
                #establishing connection with NetSim at the start of the episode
                # server_address = (socket.gethostbyname(socket.gethostname()), self.port)
                self.server_address = (socket.gethostbyname(socket.gethostname()), self.port)
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket.connect(self.server_address)
                print("Connected to the server")
                break
            except:
                continue
        
        request_value = 0
        msgType = 0
        packed_value = struct.pack(">II",msgType, request_value)
        #send a request value to NetSim
        bytes_Sent = self.client_socket.send(packed_value)
        #recieve SINRs from NetSim
        data = self.client_socket.recv(self.bytes_to_receive)
        format_string = f'>{2*self.num_UEs+1}d'
        unpacked_data = struct.unpack(f'>{format_string[1:]}', data)

        # self.state = [unpacked_data[i] for i in range(self.num_UEs)]
        self.state = unpacked_data[0:self.num_UEs]
        # throughputs_NETSIM = [unpacked_data[i] for i in range(6,12)]
        # throughputs_NETSIM = unpacked_data[self.num_UEs:2*self.num_UEs]

    def step(self, action):
        self.it+=1
        # print("iteration ", self.it)
        # Apply action
        delta_p = self.action_to_power_adjustments(action)
        self.current_gNB_powers = [p + dp for p, dp in zip(self.current_gNB_powers, delta_p)]
        self.current_gNB_powers = [max(27, min(46, p)) for p in self.current_gNB_powers]
        
        done = False
        # Additional info
        info = {}
        # # Reward calculation
        # sinrs_current = [self.dB_to_linear(i) for i in self.state]
        # throughputs_Mbps_current =[ (self.B_MHz * np.log2(1 + i)) for i in sinrs_current]
        # reward_current = np.sum(throughputs_Mbps_current)
        # reward = reward_current
        
        # # Simulate environment response tp get next state
        # next_state= self.simulate_environment()
        # self.total_rewards.append(reward)

        #calculating the throughput values from the received sinrs 
        # sinr_linear = [self.dB_to_linear(i) for i in self.state]
        # spectral_eff = [np.log2(1 + i) for i in sinr_linear]
        # spectral_eff_NETSIM = [self.get_spectral_eff(i) for i in spectral_eff]
        # throughputs_Mbps = [(self.BW_MHz * i * self.ue_divider * self.dl_ul_ratio * self.overhead_multiplier * self.app_by_phy) for i in spectral_eff_NETSIM]
        # reward = sum(throughputs_Mbps)

        flag, next_sinr_values, throughputs_NETSIM, reward = self.NETSIM_interface() 
        # print(next_sinr_values)
        if flag == 0:
            # next_sinr_values = sinr_values
            # print("no flag received")
            print("flag 0 at itr",self.it)
            return self.state, self.total_rewards[-1], done, False,info
        # Update state
        self.state = next_sinr_values
        self.total_rewards.append(reward)
        
        
        # print("iteration end")
        return self.state, reward, done, False,info

    def action_to_power_adjustments(self, action_index):
        delta_p = []
        for i in range(self.num_gNBs):
            delta_p.append(self.adjustments[action_index % len(self.adjustments)])
            action_index //= len(self.adjustments)
        return delta_p
    
    def get_spectral_eff(self,spectral_eff):
        for i in range(len(self.spectral_eff_list)-1,-1,-1):
            if(self.spectral_eff_list[i] <= spectral_eff):
                return self.spectral_eff_list[i]
        return 0
    # Simulating the environment every step
    # def simulate_environment(self):
    #     sinrs_dB = np.zeros(self.num_UEs)

    #     for ue in range(self.num_UEs):
    #         received_power_dBm = self.current_gNB_powers[self.associated_gNB[ue]] - self.path_losses_matrix[self.associated_gNB[ue], ue]
    #         sum_interference_power_linear = 0
    #         for gnb in range(self.num_gNBs):
    #             if gnb != self.associated_gNB[ue]:
    #                 interference_power_dBM = self.current_gNB_powers[gnb] - self.path_losses_matrix[gnb, ue]
    #                 interference_power_linear = self.dB_to_linear(interference_power_dBM)
    #                 sum_interference_power_linear += interference_power_linear
    #         thermal_noise_linear = self.dB_to_linear(self.thermal_noise)
    #         sinrs_dB[ue] = received_power_dBm + 10 * np.log10(-np.log(np.random.uniform(0, 1))) - 10 * np.log10(sum_interference_power_linear + thermal_noise_linear)

    #     return sinrs_dB
    
    # def connectreset(self):
    #     server_address = (socket.gethostbyname(socket.gethostname()), self.port)
    #     self.client_socket.connect(server_address)
    #     print("Connected to the server")

    

    def NETSIM_interface(self):
        
        dummy_array = np.zeros((self.num_UEs))

        msgType = 1
        # Construct the format string dynamically
        format_string = f'>{len(self.current_gNB_powers)}d'
        packed_data = struct.pack(f'>I{format_string[1:]}', msgType, *self.current_gNB_powers)

        try: 
            #sending a header value indicating that NetSim should recieve the gNB powers
            self.client_socket.send(packed_data)
        except:
            return 0,dummy_array,dummy_array,0
        
        try:
            #receive an acknowledgement message from NetSim
            data = self.client_socket.recv(4)
        except:
            return 0,dummy_array,dummy_array,0
            

        unpacked_data = struct.unpack('>I', data)

        msgType = 0
        request_value = 2
        packed_value = struct.pack(">II", msgType, request_value)

        try:
            #sending request to NetSim to send over new state and the reward
            self.client_socket.send(packed_value)
        except:
            return 0,dummy_array,dummy_array,0
        

        try:
            #receive the requested data
            data = self.client_socket.recv(self.bytes_to_receive)
            # print(data)
        except:
            return 0,dummy_array,dummy_array,0

        # unpacked_data = struct.unpack('>ddddddddddddd', data)
        # new_snirs = [unpacked_data[i] for i in range(self.num_UEs)]
        # new_throughputs = [unpacked_data[i] for i in range(6,12)]

        format_string = f'>{2*self.num_UEs+1}d'
        unpacked_data = struct.unpack(f'>{format_string[1:]}', data)
        new_snirs = unpacked_data[0:self.num_UEs]
        new_throughputs = unpacked_data[self.num_UEs:2*self.num_UEs]

        return 1, new_snirs, new_throughputs, unpacked_data[2*self.num_UEs]

    #can be omitted
    def render(self, mode='human'):
        if mode == 'human':
            print(f"Last total reward: {self.total_rewards[-1]}")
        elif mode == 'rgb_array':
            pass
        else:
            raise ValueError(f"Unknown render mode: {mode}")
        return None
    
    def resetcon(self):
        self.client_socket.close()
    # def sinr_reset(self): #Resets the sinr value at start of episode
    #     sinr_values = np.zeros(self.num_UEs)
    #     for ue in range(self.num_UEs):
    #         received_power_dBm = self.current_gNB_powers[self.associated_gNB[ue]] - self.path_losses_matrix[self.associated_gNB[ue], ue]
            
    #         sum_interference_power_linear = 0
    #         for gnb in range(self.num_gNBs):
    #             if gnb != self.associated_gNB[ue]:
    #                 interference_power_dBM = self.current_gNB_powers[gnb] - self.path_losses_matrix[gnb,ue]
    #                 interference_power_linear = self.dB_to_linear(interference_power_dBM)
    #                 sum_interference_power_linear += interference_power_linear 

    #         thermal_noise_linear = self.dB_to_linear(self.thermal_noise)

    #         sinr_values[ue] = received_power_dBm + 10*np.log10(-np.log(np.random.uniform(0,1))) - 10*np.log10(sum_interference_power_linear+thermal_noise_linear)

    #     self.state = sinr_values
        # print(self.state)

    def dB_to_linear(self, dBValue):
        power = (dBValue/10)
        linearValue = 10**(power)
        return linearValue

    # def log_pathloss(self,d):
    #     distance_term = 10*(self.eta_service)*(np.log10(d/self.d0))
    #     return self.const_term+distance_term

# env=gym.make('3gnb_6ue-v0')
# print("check env begin")
# check_env(env.unwrapped)
# print("check env finish \n")
# obs=env.reset()[0]
# for i in range(10):
#     rand_action=env.action_space.sample()
#     st,rew,done,_,_ = env.step(rand_action)
#     print(st)