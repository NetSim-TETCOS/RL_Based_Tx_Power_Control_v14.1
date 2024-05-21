import numpy as np
import matplotlib.pyplot as plt
import socket
import struct
import logging
import pandas as pd
import time
import os

parent_dir = os.getcwd()

directory_plots = "plots"
path_plots = os.path.join(parent_dir, directory_plots)
print(os.path.isdir(path_plots))
if(not os.path.isdir(path_plots)):
  os.mkdir(path_plots)

directory_logs = "logs"
path_logs = os.path.join(parent_dir, directory_logs)
print(os.path.isdir(path_logs))
if(not os.path.isdir(path_logs)):
    os.mkdir(path_logs)

#registering the start time of the simulation
start_time = time.time()

num_gNBs = 3
num_UEs = 6
BW_MHz = 50
adjustments = [-3, -1, 0, 1, 3]  # Possible adjustments
num_UE_buckets = 4
num_gNB_buckets = len(adjustments)
num_states = num_UE_buckets ** num_UEs  # 4096 states
num_actions = num_gNB_buckets ** num_gNBs  # 125 actions
q_table = np.zeros((num_states, num_actions))
gNB_powers = [40, 40, 40]
num_episodes = (int)(input("Enter the number of episodes\n"))
# num_episodes = 500
epsilon = 0.2
gamma = 0.9
alpha = 0.3
bits_to_receive = (num_UEs+1)*8

total_rewards_per_episode = []

dl_ul_ratio = 4/5
overhead_multiplier = 0.86
app_by_phy = 0.70
ue_divider = 0.5

logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger('GNB_Powers_Logger')
sinr_logger = logging.getLogger('SINR_Logger')
all_values_logger = logging.getLogger('ALL_VALUE_Logger')

# Create a CSV file handler for logging gNB powers
log_filename = os.path.join(path_logs, "gnb_powers_log.csv")
gnb_powers_csv_handler = logging.FileHandler(log_filename, mode='w')
logger.addHandler(gnb_powers_csv_handler)

# Create a CSV file handler for logging SINR values
log_filename = os.path.join(path_logs, "sinr_values_log.csv")
sinr_csv_handler = logging.FileHandler(log_filename, mode='w')
sinr_logger.addHandler(sinr_csv_handler)

# Create a CSV file handler for logging throuhgputs and sinr values at each iteration 
log_filename = os.path.join(path_logs, "all_values_log.csv")
all_values_csv_handler = logging.FileHandler(log_filename, mode='w')
all_values_logger.addHandler(all_values_csv_handler)


# Write headers to the CSV files without newline characters
gnb_powers_csv_handler.stream.write('gNB_Power_1,gNB_Power_2,gNB_Power_3\n')
sinr_csv_handler.stream.write('UE1_Throughput,UE2_Throughput,UE3_Throughput,UE4_Throughput,UE5_Throughput,UE6_Throughput\n')
all_values_csv_handler.stream.write('UE1_Throughput,UE2_Throughput,UE3_Throughput,UE4_Throughput,UE5_Throughput,UE6_Throughput,Sum_Throughput,UE1_SINR,UE2_SINR,UE3_SINR,UE4_SINR,UE5_SINR,UE6_SINR,Episode, Iteration\n')

#CQI table 
spectral_eff_list = [0.1523, 0.3770, 0.8770, 1.4766, 1.9141, 2.4063, 2.7305, 3.3223, 3.9023, 4.5234, 5.1152, 5.5547, 6.2266, 6.9141, 7.4063]


#functions to log values in the respective files
def log_gnb_powers(gNB_powers):
    logger.info(",".join(map(str, gNB_powers)))

def log_sinr_values(sinr_values):
    sinr_logger.info(",".join(map(str, sinr_values)))

def log_all_values(all_values):
    all_values_logger.info(",".join(map(str, all_values)))


# Function to map an action index to power adjustments
def action_to_power_adjustments(action_index):
    
    delta_p = []
    for i in range(num_gNBs):
        delta_p.append(adjustments[action_index % num_gNB_buckets])
        action_index //= num_gNB_buckets
    return delta_p


#discretize the states based on the 25th, 50th, 75th percentiles of the sinr data
def get_state(sinr_values):
    state = 0
    percentiles = [-11.550987805408719, -5.915791373545139, -0.0417750795523901]

    for sinr in sinr_values:
        if sinr < percentiles[0]:
            state = state * 4 + 0
        elif sinr < percentiles[1]:
            state = state * 4 + 1
        elif sinr < percentiles[2]:
            state = state * 4 + 2
        else :
            state = state * 4 + 3
    return state

def dB_to_linear(dBValue):
    power = (dBValue/10)
    linearValue = 10**(power)
    return linearValue

def get_spectral_eff(spectral_eff):
    for i in range(len(spectral_eff_list)-1,-1,-1):
        if(spectral_eff_list[i] <= spectral_eff):
            return spectral_eff_list[i]
    return 0

#receive data from NetSim at the start of the episode
def episode_Start():
    request_value = 0
    msgType = 0
    packed_value = struct.pack(">II",msgType, request_value)

    #send a request value to NetSim
    bytes_Sent = client_socket.send(packed_value)
    
    #recieve SINRs from NetSim
    data = client_socket.recv(bits_to_receive)

    return data

def NETSIM_interface(gNB_powers):
    
    dummy_array = np.zeros((num_UEs))

    msgType = 1
    packed_data = struct.pack('>Iddd', msgType, gNB_powers[0], gNB_powers[1], gNB_powers[2])
    
    try: 
        #sending a header value indicating that NetSim should recieve the gNB powers
        client_socket.send(packed_data)
    except:
      return 0,dummy_array,0
    
    try:
        #receive an acknowledgement message from NetSim
        data = client_socket.recv(4)
    except:
        return 0,dummy_array,0
        

    unpacked_data = struct.unpack('>I', data)

    msgType = 0
    request_value = 2
    packed_value = struct.pack(">II", msgType, request_value)

    try:
        #sending request to NetSim to send over new state and the reward
        client_socket.send(packed_value)
    except:
        return 0,dummy_array,0
    

    try:
        #receive the requested data
        data = client_socket.recv(bits_to_receive)
    except:
        return 0,dummy_array,0

    unpacked_data = struct.unpack('>ddddddd', data)
    new_snirs = [unpacked_data[i] for i in range(num_UEs)]

    return 1, new_snirs, unpacked_data[num_UEs]


for episode in range(num_episodes):

    total_reward = 0  # Initialize total reward for this episode
    
    while True:
        try:
            #establishing connection with NetSim at the start of the episode
            server_address = (socket.gethostbyname(socket.gethostname()), 12345)
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect(server_address)
            print("Connected to the server")
            break
        except:
            continue
    
    #receive data at the start of the episode
    data = episode_Start()

    unpacked_data = struct.unpack('>ddddddd', data)

    sinr_values = [unpacked_data[i] for i in range(num_UEs)]

    state = get_state(sinr_values)

     # Reset gNB powers to initial state at the start of each episode
    current_gNB_powers = [40, 40, 40]
    steps_per_episode = 1000

    for t in range(steps_per_episode):  # Limit steps in each episode
        if np.random.rand() < epsilon:
            action_index = np.random.randint(num_actions)  # Explore
        else:
            action_index = np.argmax(q_table[state])  # Exploit

          # Update current_gNB_powers based on the action taken
        delta_p = action_to_power_adjustments(action_index)
        current_gNB_powers = [p + dp for p, dp in zip(current_gNB_powers, delta_p)]

        # Restrict gNB powers between 27 to 46 dBm
        current_gNB_powers = [max(27, min(46, p)) for p in current_gNB_powers]
        
        #recieve next state and reward from NetSim
        flag, next_sinr_values, reward = NETSIM_interface(current_gNB_powers)

        if flag == 0:
            next_sinr_values = sinr_values

        next_state = get_state(next_sinr_values)
        total_reward += reward

        # Q-learning update
        q_table[state, action_index] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action_index])
        
        #calculating the throughput values from the received sinrs 
        sinr_linear = [dB_to_linear(i) for i in sinr_values]
        spectral_eff = [np.log2(1 + i) for i in sinr_linear]
        spectral_eff_NETSIM = [get_spectral_eff(i) for i in spectral_eff]
        throughputs_Mbps = [(BW_MHz * i * ue_divider * dl_ul_ratio * overhead_multiplier * app_by_phy) for i in spectral_eff_NETSIM]

        log_all_values([throughputs_Mbps[0],throughputs_Mbps[1],throughputs_Mbps[2],throughputs_Mbps[3],throughputs_Mbps[4],throughputs_Mbps[5], reward, next_sinr_values[0],next_sinr_values[1],next_sinr_values[2],next_sinr_values[3],next_sinr_values[4],next_sinr_values[5], episode, t])
        
        #logging the throughputs and gNB powers in the last episode
        if episode == num_episodes-1:
            log_sinr_values(throughputs_Mbps)
            log_gnb_powers(current_gNB_powers)


        state = next_state
        sinr_values = next_sinr_values  
        print(f"Episode {episode}, Time_Step {t}")

    total_rewards_per_episode.append(total_reward/steps_per_episode)

    # if episode % 1 == 0:
    print(f"Episode {episode}: Completed")
    client_socket.close()

#plotting the average rewards in each episode
plt.plot(total_rewards_per_episode)
plt.title("Average rewards per episode")
plt.xlabel("Episodes")
plt.ylabel("Average Sum Throuhghput(Mbps)")

plot_filename = os.path.join(path_plots, "Sum_Throughputs.png")
plt.savefig(plot_filename)
plt.show()

log_filename = os.path.join(path_logs, "gnb_powers_log.csv")
gnb_DF = pd.read_csv(log_filename)

log_filename = os.path.join(path_logs, "sinr_values_log.csv")
sinr_DF = pd.read_csv(log_filename)

# Delete the last row from the DataFrame
gnb_DF.drop(gnb_DF.tail(1).index, inplace=True)
sinr_DF.drop(sinr_DF.tail(1).index, inplace=True)

gnb1 = gnb_DF["gNB_Power_1"]
gnb2 = gnb_DF["gNB_Power_2"]
gnb3 = gnb_DF["gNB_Power_3"]


#plotting the Tx powers of each gNB over the last episode
plt.figure(figsize=(10,6))
plt.plot(gnb1)
plt.plot(gnb2)
plt.plot(gnb3)
plt.legend(['gNB1','gNB2','gNB3'])
plt.title("gNB power vs. Time. Optimal Policy")
plt.xlabel("Iterations")
plt.ylabel("gNB power (dBm)")
plot_filename = os.path.join(path_plots, "gNB_Powers.png")
plt.savefig(plot_filename)
plt.show()

#plotting the UE throughputs in separate plots
plt.figure(figsize=(10,6))
plt.grid(True)
plt.title("Individual UE throughput (Mbps) vs. time. Optimal Policy")
plt.xlabel("Iterations")
plt.ylabel("Individual UE throughput (Mbps)")

plt.plot(sinr_DF["UE1_Throughput"])
plot_filename = os.path.join(path_plots, "Individual_UE_throughput_1.png")
plt.savefig(plot_filename)
plt.show()
plt.figure(figsize=(10,6))
plt.grid(True)
plt.title("Individual UE throughput (Mbps) vs. time. Optimal Policy")
plt.xlabel("Iterations")
plt.ylabel("Individual UE throughput (Mbps)")
plt.plot(sinr_DF["UE2_Throughput"])
plot_filename = os.path.join(path_plots, "Individual_UE_throughput_2.png")
plt.savefig(plot_filename)
plt.show()
plt.figure(figsize=(10,6))
plt.grid(True)
plt.title("Individual UE throughput (Mbps) vs. time. Optimal Policy")
plt.xlabel("Iterations")
plt.ylabel("Individual UE throughput (Mbps)")
plt.plot(sinr_DF["UE3_Throughput"])
plot_filename = os.path.join(path_plots, "Individual_UE_throughput_3.png")
plt.savefig(plot_filename)
plt.show()
plt.figure(figsize=(10,6))
plt.grid(True)
plt.title("Individual UE throughput (Mbps) vs. time. Optimal Policy")
plt.xlabel("Iterations")
plt.ylabel("Individual UE throughput (Mbps)")
plt.plot(sinr_DF["UE4_Throughput"])
plot_filename = os.path.join(path_plots, "Individual_UE_throughput_4.png")
plt.savefig(plot_filename)
plt.show()
plt.figure(figsize=(10,6))
plt.grid(True)
plt.title("Individual UE throughput (Mbps) vs. time. Optimal Policy")
plt.xlabel("Iterations")
plt.ylabel("Individual UE throughput (Mbps)")
plt.plot(sinr_DF["UE5_Throughput"])
plot_filename = os.path.join(path_plots, "Individual_UE_throughput_5.png")
plt.savefig(plot_filename)
plt.show()
plt.figure(figsize=(10,6))
plt.grid(True)
plt.title("Individual UE throughput (Mbps) vs. time. Optimal Policy")
plt.xlabel("Iterations")
plt.ylabel("Individual UE throughput (Mbps)")
plt.plot(sinr_DF["UE6_Throughput"])
plot_filename = os.path.join(path_plots, "Individual_UE_throughput_6.png")
plt.savefig(plot_filename)
plt.show()


#printing the total time taken for the RL simulation
print(f"Process finished --- {(time.time()-start_time)/60} minutes ---")