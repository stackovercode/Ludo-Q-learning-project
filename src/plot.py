import sys
import numpy as np
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
from pandas.plotting import parallel_coordinates
from pandas.plotting import scatter_matrix
import matplotlib.gridspec as gridspec


def rolling_average(data, window):
    cumsum, moving_aves = [0], []
    for i, x in enumerate(data, 1):
        cumsum.append(cumsum[i-1] + x)
        if i >= window:
            moving_ave = (cumsum[i] - cumsum[i-window]) / window
            moving_aves.append(moving_ave)
    return moving_aves


def exponential_average(data, alpha):
    ema = [data[0]]  # starts from the first data point
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[i-1])
    return ema


parameters = np.load('/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project/src/data/parameters.npy', allow_pickle=True)
data = np.load('/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project/src/data/data.npy',  allow_pickle=True)
win_rate_vec = data[0]
actions_per_game = data[1]
games_wins = data[2]
average_win_rates = data[3]

while actions_per_game[0] == 0:
    actions_per_game = actions_per_game[1:]

boltzmann_temperature = parameters[0]
discount_factor = parameters[1]
learning_rate = parameters[2]

highest_win_rate = 0
best_index = [0,0,0]
best_data = []
temp_data = []
temp_win_rates = []


# Your default parameters
default_learning_rate = 0.325
default_discount_factor = 0.175
default_boltzmann_temperature = 0.125

boltzmann_temperature_stored = []
discount_factor_stored = []
learning_rate_stored = []


# Placeholder for your win rates
win_rate_lr = []
combindedata1 = []
data1 = []

win_rate_df = []
combindedata2 = []
data2 = [] 

win_rate_bt = []
combindedata3 = []
data3 = []

#win_rates_by_lr = {lr: [] for lr in learning_rate}


avg_win_rate = np.zeros((len(boltzmann_temperature), len(discount_factor), len(learning_rate)))
data_long = pd.DataFrame(columns=['Boltzmann Temperature', 'Discount Factor', 'Learning Rate', 'Win Rate'])

for (BTidx, BTval), (DFidx, DFval), (LRidx, LRval) in itertools.product(enumerate(boltzmann_temperature), enumerate(discount_factor), enumerate(learning_rate)):
    #Append this combination and its win rate to the DataFrame
    data_long = data_long.append({
        'Boltzmann Temperature': BTval,
        'Discount Factor': DFval,
        'Learning Rate': LRval,
        'Win Rate': avg_win_rate[BTidx][DFidx][LRidx]
    }, ignore_index=True)
    avg_win_rate[BTidx][DFidx][LRidx] = np.mean(win_rate_vec[BTidx][DFidx][LRidx])
    temp_win_rate = np.sum(win_rate_vec[BTidx][DFidx][LRidx]) / len(win_rate_vec[BTidx][DFidx][LRidx])
    temp_win_rates.append(temp_win_rate)
    if DFval == default_discount_factor and BTval == default_boltzmann_temperature:
        data1 = exponential_average(win_rate_vec[BTidx][DFidx][LRidx], 0.5)
        combindedata1.append(data1)
        learning_rate_stored.append(LRval)
        win_rate_lr.append(temp_win_rate)
    if LRval == default_learning_rate and BTval == default_boltzmann_temperature:
        data2 = exponential_average(win_rate_vec[BTidx][DFidx][LRidx], 0.5)
        combindedata2.append(data2)
        discount_factor_stored.append(DFval)
        win_rate_df.append(temp_win_rate)
    if LRval == default_learning_rate and DFval == default_discount_factor:
        data3 = exponential_average(win_rate_vec[BTidx][DFidx][LRidx], 0.5)
        combindedata3.append(data3)
        boltzmann_temperature_stored.append(BTval)
        win_rate_bt.append(temp_win_rate)

    if temp_win_rate > highest_win_rate :
        highest_win_rate = temp_win_rate
        temp_data = exponential_average(win_rate_vec[BTidx][DFidx][LRidx], 0.5)
        best_data = win_rate_vec[BTidx][DFidx][LRidx]
        best_index = [BTval, DFval, LRval]
        
        
# ############ Compara paramters combined plot ############         
# fig, axs = plt.subplots(3, 1, figsize=(8, 18))  # 3 rows, 1 column
# # Plot Learning Rate
# for i, data in enumerate(combindedata1):
#     axs[0].plot(range(1, len(data1) + 1), [element * 100 for element in data], label=f'LR={learning_rate_stored[i]}')
# axs[0].set_ylabel('Win rate [%] - Learning Rate', fontsize=14)
# axs[0].legend(loc=4, fontsize=12)
# #axs[0].set_title('Learning Rate')

# # Plot Discount Factor
# for i, data in enumerate(combindedata2):
#     axs[1].plot(range(1, len(data2) + 1), [element * 100 for element in data], label=f'DF={discount_factor_stored[i]}')
# axs[1].set_ylabel('Win rate [%] - Discount Factor', fontsize=14)
# axs[1].legend(loc=4, fontsize=14)
# #axs[1].set_title('Discount Factor')

# # Plot Boltzmann Temperature
# for i, data in enumerate(combindedata3):
#     axs[2].plot(range(1, len(data3) + 1), [element * 100 for element in data], label=f'BT={boltzmann_temperature_stored[i]}')
# axs[2].set_xlabel('Number of games', fontsize=14)
# axs[2].set_ylabel('Win rate [%] - Boltzmann Temperature', fontsize=14)
# axs[2].legend(loc=4, fontsize=14)
# #axs[2].set_title('Boltzmann Temperature')
# # Save and display the figure
# plt.tight_layout()
# plt.savefig('/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project/src/images/comparison_parameters.png', bbox_inches='tight')
# plt.show()        
      
# # ############ Compara paramters ############        
# label_fontsize = 14
# title_fontsize = 16
# ticks_fontsize = 12
# legend_fontsize = 11

# # Plot Learning Rate
# fig, axs = plt.subplots()
# for i, data in enumerate(combindedata1):
#     axs.plot(range(1, len(data1) + 1), [element * 100 for element in data], label=f'LR={learning_rate_stored[i]}')
# axs.set_xlabel('Number of games', fontsize=label_fontsize)
# axs.set_ylabel('Win rate [%]', fontsize=label_fontsize)
# plt.xticks(fontsize=ticks_fontsize)
# plt.yticks(fontsize=ticks_fontsize)
# axs.legend(loc=4, fontsize=legend_fontsize)
# axs.set_title('Learning Rate', fontsize=title_fontsize)
# plt.savefig('/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project/src/images/winrate_LR.png', bbox_inches='tight')
# plt.show()

# # Plot Discount Factor
# fig, axs = plt.subplots()
# for i, data in enumerate(combindedata2):
#     axs.plot(range(1, len(data2) + 1), [element * 100 for element in data], label=f'DF={discount_factor_stored[i]}')
# axs.set_xlabel('Number of games', fontsize=label_fontsize)
# axs.set_ylabel('Win rate [%]', fontsize=label_fontsize)
# plt.xticks(fontsize=ticks_fontsize)
# plt.yticks(fontsize=ticks_fontsize)
# axs.legend(loc=4, fontsize=legend_fontsize)
# axs.set_title('Discount Factor', fontsize=title_fontsize)
# plt.savefig('/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project/src/images/winrate_DF.png', bbox_inches='tight')
# plt.show()

# # Plot Boltzmann Temperature
# fig, axs = plt.subplots()
# for i, data in enumerate(combindedata3):
#     axs.plot(range(1, len(data3) + 1), [element * 100 for element in data], label=f'BT={boltzmann_temperature_stored[i]}')
# axs.set_xlabel('Number of games', fontsize=label_fontsize)
# axs.set_ylabel('Win rate [%]', fontsize=label_fontsize)
# plt.xticks(fontsize=ticks_fontsize)
# plt.yticks(fontsize=ticks_fontsize)
# axs.legend(loc=4, fontsize=legend_fontsize)
# axs.set_title('Boltzmann Temperature', fontsize=title_fontsize)
# plt.savefig('/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project/src/images/winrate_BT.png', bbox_inches='tight')
# plt.show()
     


# # # ############ PLOT 1 - Only win rate ############
# label_fontsize = 14
# legend_fontsize = 11

# print(highest_win_rate, best_index)
# fig, axs = plt.subplots()
# axs.set_xlabel('Number of games', fontsize=label_fontsize)
# axs.set_ylabel('Win rate [%]', fontsize=label_fontsize)
# axs.plot(range(1, len(best_data) + 1), [element * 100 for element in best_data], label='Raw win rate',  color='blue')
# axs.plot(range(1, len(temp_data) + 1), [element * 100 for element in temp_data], linewidth=2, label='Smooth win rate',  color='orange')
# #axs.plot(range(1, len(temp_data2) + 1), [element * 100 for element in temp_data2], linewidth=2, label='Rolling average',  color='red')

# axs.legend(loc=4, fontsize=legend_fontsize)
# plt.savefig('/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project/src/images/Plot_1_Best_winrate.png', bbox_inches='tight')
# plt.show()

# # ############ PLOT 1.1 - Only win rate ############ ####### ADDITION OF SVG.
label_fontsize = 14
legend_fontsize = 11

print(highest_win_rate, best_index)
fig, axs = plt.subplots()
axs.set_xlabel('Number of games', fontsize=label_fontsize)
axs.set_ylabel('Win rate [%]', fontsize=label_fontsize)
axs.plot(range(1, len(best_data) + 1), [element * 100 for element in best_data], label='Raw win rate',  color='blue')
axs.plot(range(1, len(temp_data) + 1), [element * 100 for element in temp_data], linewidth=2, label='Smooth win rate',  color='orange')
#axs.plot(range(1, len(temp_data2) + 1), [element * 100 for element in temp_data2], linewidth=2, label='Rolling average',  color='red')

axs.legend(loc=4, fontsize=legend_fontsize)
plt.savefig('/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project/src/images/Plot_1_Best_winrate.svg', format='svg', dpi=1200)
plt.show()



# # ############ PLOT 2 - used to compare ############
# average_win_rates_data = exponential_average(average_win_rates, 0.3)
# # create a list of episode numbers matching the length of your win rates
# episodes = list(range(1, len(average_win_rates_data) + 1))
# ## create the scatter plot
# plt.scatter(episodes, average_win_rates_data)
# plt.plot(episodes, exponential_average(average_win_rates, 0.1), label='Average Win Rate', linewidth = 2, color='red')
# # set the title and labels
# plt.title('Average Win Rates over Episodes')
# plt.xlabel('Episode')
# plt.ylabel('Average Win Rate')
# # Set the y-axis limits and step size
# #plt.ylim([0, 0.6])
# plt.ylim([0.4, 0.9])
# plt.yticks([i/10 for i in range(0,10)])
# plt.savefig('/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project/src/images/Plot_2_To_compare.png', bbox_inches='tight')
# plt.show()


# # ############ PLOT 3 - Visualizing Actions per game ############
games_played_raw = list(range(1, len(actions_per_game) + 1))
# # Generate x values for moving average data
actions_per_game_data = rolling_average(actions_per_game, 15)
games_played_ma = list(range(1, len(actions_per_game_data) + 1))  
# plt.figure(figsize=(10, 6))
# # Plot raw data
# plt.plot(games_played_raw, actions_per_game, label='Actions per game')
# # Plot moving average data
# plt.plot(games_played_ma, actions_per_game_data, label='Moving Average Actions per game')
# plt.xlabel('Number of Games Played')
# plt.ylabel('Actions per Game')
# plt.title('Number of Actions per Game over Time')
# plt.legend()
# plt.grid(False)
# plt.savefig('/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project/src/images/Plot_3_Visualizing_Actions.png', bbox_inches='tight')
# plt.show()


# ############ Plot 4 - OVERALL induvidual plots ############
# games_played_raw = list(range(1, len(actions_per_game) + 1))
# actions_per_game_data = rolling_average(actions_per_game, 15)
# games_played_ma = list(range(1, len(actions_per_game_data) + 1))

# ############ Plot 1 - Win rate over games ############
# plt.figure(figsize=(5, 5))
# plt.xlabel('Number of games', fontsize=12)
# plt.ylabel('Win rate [%]', fontsize=12)
# plt.plot(range(1, len(best_data) + 1), [element * 100 for element in best_data], label='Raw win rate',  color='blue')
# plt.plot(range(1, len(temp_data) + 1), [element * 100 for element in temp_data], linewidth=2, label='Smooth win rate',  color='orange')
# plt.legend(loc=4, fontsize=12)
# plt.show()

# ############ Plot 2 - Actions per game over time ############
# plt.figure(figsize=(5, 5))
# plt.plot(games_played_raw, actions_per_game, label='Actions per game', color='blue')
# plt.plot(games_played_ma, actions_per_game_data, label='Moving Average Actions per game', color='orange')
# plt.xlabel('Number of Games Played', fontsize=14)
# plt.ylabel('Actions per Game', fontsize=14)
# plt.title('Actions per game over Time', fontsize=16)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# #plt.legend(prop={'size': 16})
# plt.savefig('/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project/src/images/Plot 2.png', bbox_inches='tight')
# plt.show()

# # ############ Plot 3 - Histogram of actions for won games ############
# games_wins_flat = np.array(games_wins).flatten()
# actions_won_games = [actions_per_game[i] for i in range(len(actions_per_game)) if games_wins_flat[i] == 1]
# actions_lost_games = [actions_per_game[i] for i in range(len(actions_per_game)) if games_wins_flat[i] == 0]

# plt.figure(figsize=(5, 5))
# plt.hist(actions_won_games, bins=50, alpha=0.5, label='Won Games', color='blue')
# plt.hist(actions_lost_games, bins=50, alpha=0.5, label='Lost Games', color='orange')
# plt.xlabel('Actions per Game', fontsize=14)
# plt.ylabel('Frequency', fontsize=14)
# plt.title('Actions per game over Result', fontsize=16)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# #plt.legend(prop={'size': 16})
# plt.legend(loc=2, fontsize=12)
# plt.savefig('/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project/src/images/Plot 3.png', bbox_inches='tight')
# plt.show()

# ############  Plot 4 - Average win rates for each parameter ############
# avg_win_rate_BT = np.mean(avg_win_rate, axis=(1,2))
# avg_win_rate_DF = np.mean(avg_win_rate, axis=(0,2))
# avg_win_rate_LR = np.mean(avg_win_rate, axis=(0,1))

# # Average win rates for each Boltzmann Temperature
# plt.figure(figsize=(5, 5))
# bars = plt.bar(boltzmann_temperature, avg_win_rate_BT, width=0.01, align='center', color='orange')
# bars[np.argmax(avg_win_rate_BT)].set_color('blue')  # set the color of the highest bar to blue
# plt.xlabel('Boltzmann Temperature')
# plt.ylabel('Average Win Rate')
# plt.show()

# # Average win rates for each Discount Factor
# plt.figure(figsize=(5, 5))
# bars = plt.bar(discount_factor, avg_win_rate_DF,  width=0.01, align='center', color='orange')
# bars[np.argmax(avg_win_rate_DF)].set_color('blue')  # set the color of the highest bar to blue
# plt.xlabel('Discount Factor')
# plt.show()

# # Average win rates for each Learning Rate
# plt.figure(figsize=(5, 5))
# bars = plt.bar(learning_rate, avg_win_rate_LR,  width=0.01, align='center', color='orange')
# bars[np.argmax(avg_win_rate_LR)].set_color('blue')  # set the color of the highest bar to blue
# plt.xlabel('Learning Rate')
# plt.show()


# ############ Plot 4 - OVERALL PLOT ############
# # Define your 2x2 main subplots
# fig = plt.figure(figsize=(15, 15))
# gs = gridspec.GridSpec(2, 2, figure=fig)
# games_played_raw = list(range(1, len(actions_per_game) + 1))
# actions_per_game_data = rolling_average(actions_per_game, 15)
# games_played_ma = list(range(1, len(actions_per_game_data) + 1))  

# ax1 = fig.add_subplot(gs[0, 0])
# ax2 = fig.add_subplot(gs[0, 1])
# ax3 = fig.add_subplot(gs[1, 0])

# # Define a 1x3 subplot grid within the fourth main subplot
# gs_inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 1])

# ax4_1 = fig.add_subplot(gs_inner[0, 0])
# ax4_2 = fig.add_subplot(gs_inner[0, 1])
# ax4_3 = fig.add_subplot(gs_inner[0, 2])

# # Plot 1: Win rate over games
# ax1.set_xlabel('Number of games', fontsize=12)
# ax1.set_ylabel('Win rate [%]', fontsize=12)
# ax1.plot(range(1, len(best_data) + 1), [element * 100 for element in best_data], label='Raw win rate',  color='blue')
# ax1.plot(range(1, len(temp_data) + 1), [element * 100 for element in temp_data], linewidth=2, label='Smooth win rate',  color='orange')
# ax1.legend(loc=4, fontsize=12)

# # Plot 2: Actions per game over time
# ax2.plot(games_played_raw, actions_per_game, label='Actions per game', color='blue')
# ax2.plot(games_played_ma, actions_per_game_data, label='Moving Average Actions per game', color='orange')
# ax2.set_xlabel('Number of Games Played')
# ax2.set_ylabel('Actions per Game')
# ax2.set_title('Number of Actions per Game over Time')
# ax2.legend()
# ax2.grid(False)


# ############ Plot 3: Histogram of actions for won games
# ############ Histogram of actions for won games
# games_wins_flat = np.array(games_wins).flatten()
# actions_won_games = [actions_per_game[i] for i in range(len(actions_per_game)) if games_wins_flat[i] == 1]
# actions_lost_games = [actions_per_game[i] for i in range(len(actions_per_game)) if games_wins_flat[i] == 0]
# ax3.hist(actions_won_games, bins=50, alpha=0.5, label='Won Games', color='blue')
# ax3.hist(actions_lost_games, bins=50, alpha=0.5, label='Lost Games', color='orange')
# ax3.set_xlabel('Actions per Game')
# ax3.set_ylabel('Frequency')
# ax3.set_title('Number of Actions per Game for Won and Lost Games')
# ax3.legend()

# ############ Plot 3.1: Histogram of actions for won games
# ############ Histogram of actions for won games
# games_wins_flat = np.array(games_wins).flatten()
# actions_won_games = [actions_per_game[i] for i in range(len(actions_per_game)) if games_wins_flat[i] == 1]
# actions_lost_games = [actions_per_game[i] for i in range(len(actions_per_game)) if games_wins_flat[i] == 0]
# ax3.hist(actions_won_games, bins=50, alpha=0.5, label='Won Games', color='blue', density=True)
# ax3.hist(actions_lost_games, bins=50, alpha=0.5, label='Lost Games', color='orange', density=True)
# ax3.set_xlabel('Actions per Game')
# ax3.set_ylabel('Probability')
# ax3.set_title('Probability Distribution of Actions per Game for Won and Lost Games')
# ax3.legend()

# ############ Plot 3.2: Histogram of actions for won games
# ############ Histogram of actions for won games
# games_wins_flat = np.array(games_wins).flatten()
# actions_won_games = [actions_per_game[i] for i in range(len(actions_per_game)) if games_wins_flat[i] == 1]
# actions_lost_games = [actions_per_game[i] for i in range(len(actions_per_game)) if games_wins_flat[i] == 0]
# sns.distplot(actions_won_games, bins=50, label='Won Games', color='blue', ax=ax3)
# sns.distplot(actions_lost_games, bins=50, label='Lost Games', color='orange', ax=ax3)
# ax3.set_xlabel('Actions per Game')
# ax3.set_ylabel('Probability')
# ax3.set_title('Probability Distribution of Actions per Game for Won and Lost Games')
# ax3.legend()


# avg_win_rate_BT = np.mean(avg_win_rate, axis=(1,2))
# # Compute average win rate for each Discount Factor
# avg_win_rate_DF = np.mean(avg_win_rate, axis=(0,2))
# # Compute average win rate for each Learning Rate
# avg_win_rate_LR = np.mean(avg_win_rate, axis=(0,1))


# ############ Plot 4  ############
# # Plot 4.1: Average win rates for each Boltzmann Temperature
# bars = ax4_1.bar(boltzmann_temperature, avg_win_rate_BT, width=0.01, align='center', color='orange')
# bars[np.argmax(avg_win_rate_BT)].set_color('blue')  # set the color of the highest bar to blue

# #ax4_1.bar(boltzmann_temperature, avg_win_rate_BT,width=0.01, align='center', color='orange')
# ax4_1.set_xlabel('Boltzmann Temperature')
# ax4_1.set_ylabel('Average Win Rate')

# # Plot 4.2: Average win rates for each Discount Factor
# bars = ax4_2.bar(discount_factor, avg_win_rate_DF,  width=0.01, align='center', color='orange')
# bars[np.argmax(avg_win_rate_DF)].set_color('blue')  # set the color of the highest bar to blue

# ax4_2.set_xlabel('Discount Factor')

# # Plot 4.3: Average win rates for each Learning Rate
# bars = ax4_3.bar(learning_rate, avg_win_rate_LR,  width=0.01, align='center', color='orange')
# bars[np.argmax(avg_win_rate_LR)].set_color('blue')  # set the color of the highest bar to blue

# ax4_3.set_xlabel('Learning Rate')


# plt.tight_layout()
# plt.savefig('/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project/src/images/Plot_4_OVERALL_PLOT.png', bbox_inches='tight')
# plt.show()


############ Plot 5: Average win rates for each Boltzmann Temperature ############
# avg_win_rate_BT = np.mean(avg_win_rate, axis=(1,2))
# avg_win_rate_DF = np.mean(avg_win_rate, axis=(0,2))
# avg_win_rate_LR = np.mean(avg_win_rate, axis=(0,1))

# fig, (ax1, ax2, ax3) = plt.subplots(3)
# # Setting fontsize for labels, title and ticks
# label_fontsize = 14
# title_fontsize = 16
# ticks_fontsize = 12
# text_fontsize = 11

# # Set y-axis range and increment
# y_min = 0.4
# y_max = 0.8
# y_increment = 0.1


# # Plot 1: Average win rates for each Boltzmann Temperature
# ax1.bar(boltzmann_temperature, avg_win_rate_BT, width=0.01, align='center', color='orange')
# ax1.set_xlabel('Boltzmann Temperature', fontsize=label_fontsize)
# #ax1.set_ylabel('Average Win Rate', fontsize=label_fontsize)
# ax1.set_title('Win Rate over each paramater', fontsize=title_fontsize)
# ax1.tick_params(axis='both', which='major', labelsize=ticks_fontsize)
# ax1.set_ylim(y_min, y_max)  # Set y-axis range
# #ax1.set_yticks(np.arange(y_min, y_max+y_increment, y_increment))  # Set y-axis increment
# ax1.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)  # Add grid

# highest_BT_index = np.argmax(avg_win_rate_BT)
# ax1.patches[highest_BT_index].set_facecolor('blue')
# ax1.text(boltzmann_temperature[highest_BT_index], y_min + 0.02*(y_max-y_min), 
#          str(round(boltzmann_temperature[highest_BT_index], 3)), color='white', ha='center', va='bottom', fontsize=text_fontsize)

# # Plot 2: Average win rates for each Discount Factor
# ax2.bar(discount_factor, avg_win_rate_DF, width=0.01, align='center', color='orange')
# ax2.set_xlabel('Discount Factor', fontsize=label_fontsize)
# ax2.set_ylabel('Average Win Rate', fontsize=label_fontsize)
# #ax2.set_title('Average Win Rate for each Discount Factor', fontsize=title_fontsize)
# ax2.tick_params(axis='both', which='major', labelsize=ticks_fontsize)
# ax2.set_ylim(y_min, y_max)  # Set y-axis range
# #ax2.set_yticks(np.arange(y_min, y_max+y_increment, y_increment))  # Set y-axis increment
# ax2.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)  # Add grid

# highest_DF_index = np.argmax(avg_win_rate_DF)
# ax2.patches[highest_DF_index].set_facecolor('blue')
# ax2.text(discount_factor[highest_DF_index], y_min + 0.02*(y_max-y_min), 
#          str(round(discount_factor[highest_DF_index], 3)), color='white', ha='center', va='bottom', fontsize=text_fontsize)

# # Plot 3: Average win rates for each Learning Rate
# ax3.bar(learning_rate, avg_win_rate_LR, width=0.01, align='center', color='orange')
# ax3.set_xlabel('Learning Rate', fontsize=label_fontsize)
# #ax3.set_ylabel('Average Win Rate', fontsize=label_fontsize)
# #ax3.set_title('Average Win Rate for each Learning Rate', fontsize=title_fontsize)
# ax3.tick_params(axis='both', which='major', labelsize=ticks_fontsize)
# ax3.set_ylim(y_min, y_max)  # Set y-axis range
# #ax3.set_yticks(np.arange(y_min, y_max+y_increment, y_increment))  # Set y-axis increment
# ax3.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)  # Add grid

# highest_LR_index = np.argmax(avg_win_rate_LR)
# ax3.patches[highest_LR_index].set_facecolor('blue')
# ax3.text(learning_rate[highest_LR_index], y_min + 0.02*(y_max-y_min), 
#          str(round(learning_rate[highest_LR_index], 3)), color='white', ha='center', va='bottom', fontsize=text_fontsize)

# #plt.tight_layout()
# plt.savefig('/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project/src/images/Plot_5', bbox_inches='tight')
# plt.show()