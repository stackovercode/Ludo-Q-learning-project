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


def moving_average(list, N):
    cumsum, moving_aves = [0], []
    for i, x in enumerate(list, 1):
        cumsum.append(cumsum[i - 1] + x)
        if i >= N:
            moving_ave = (cumsum[i] - cumsum[i - N]) / N
            moving_aves.append(moving_ave)
    return moving_aves

def exponential_moving_average(list, alpha):
    ema = [list[0]]  # starts from the first data point
    for i in range(1, len(list)):
        ema.append(alpha * list[i] + (1 - alpha) * ema[i-1])
    return ema


parameters = np.load('/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project/src/data/parameters.npy', allow_pickle=True)
data = np.load('/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project/src/data/data.npy',  allow_pickle=True)
win_rate_vec = data[0]
actions_per_game = data[1]
while actions_per_game[0] == 0:
    actions_per_game = actions_per_game[1:]
games_wins = data[2]

boltzmann_temperature = parameters[0]
discount_factor = parameters[1]
learning_rate = parameters[2]

highest_win_rate = 0
best_index = [0,0,0]
best_data = []
temp_data = []

avg_win_rate = np.zeros((len(boltzmann_temperature), len(discount_factor), len(learning_rate)))

################## 3D scatter plot
## Initialize an empty DataFrame to hold your flattened data
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
    if temp_win_rate > highest_win_rate :
        highest_win_rate = temp_win_rate
        temp_data = exponential_moving_average(win_rate_vec[BTidx][DFidx][LRidx], 0.5)
        best_data = win_rate_vec[BTidx][DFidx][LRidx]
        best_index = [BTval, DFval, LRval]


#parallel_coordinates(data_long, 'Win Rate', color=sns.color_palette('viridis', n_colors=len(data_long['Win Rate'].unique())))


# sns.pairplot(data_long, vars=['Boltzmann Temperature', 'Discount Factor', 'Learning Rate'], hue='Win Rate', palette='viridis')

# scatter_matrix(data_long, alpha=0.2, figsize=(6, 6), diagonal='hist')

# # Now that the data is in a long format, we can create a 3D scatter plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Create a scatter plot with the parameter values as coordinates and color representing win rate
# sc = ax.scatter(data_long['Boltzmann Temperature'], data_long['Discount Factor'], data_long['Learning Rate'], c=data_long['Win Rate'], cmap='viridis')

# # Add a color bar and labels
# plt.colorbar(sc)
# ax.set_xlabel('Boltzmann Temperature')
# ax.set_ylabel('Discount Factor')
# ax.set_zlabel('Learning Rate')
# plt.title('Win Rate by Parameter Combination')

# # Show the plot
#plt.show()

# for i, LRval in enumerate(learning_rate):
#     avg_win_rate_2D = avg_win_rate[:, :, i]
    
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(avg_win_rate_2D, xticklabels=boltzmann_temperature, yticklabels=discount_factor, cmap='viridis')
#     plt.xlabel('Boltzmann Temperature')
#     plt.ylabel('Discount Factor')
#     plt.title(f'Learning Rate = {LRval}')
#     plt.show()

# Generate a pair plot
# sns.pairplot(data_long, hue='Win Rate')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # # Generate a 3D scatter plot
# sc = ax.scatter(data_long['Boltzmann Temperature'], data_long['Discount Factor'], data_long['Learning Rate'], c=data_long['Win Rate'], cmap='viridis')
# plt.colorbar(sc)
# ax.set_xlabel('Boltzmann Temperature')
# ax.set_ylabel('Discount Factor')
# ax.set_zlabel('Learning Rate')
# plt.title('Win Rate by Parameter Combination')
# plt.show()

######################


print(highest_win_rate, best_index)
# fig, axs = plt.subplots()
# axs.set_xlabel('Number of games', fontsize=12)
# axs.set_ylabel('Win rate [%]', fontsize=12)
# axs.plot(range(1, len(best_data) + 1), [element * 100 for element in best_data], label = 'Raw win rate')
# axs.plot(range(1, len(temp_data) + 1), [element * 100 for element in temp_data], linewidth = 3, label = 'Smooth win rate')
# axs.legend(loc=4, fontsize=12)

# plt.savefig('/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project/src/images/Best_winrate2.png', bbox_inches='tight')

# plt.show()

games_played_raw = list(range(1, len(actions_per_game) + 1))

# Generate x values for moving average data
actions_per_game_data = moving_average(actions_per_game, 15)
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
# plt.show()

# Histogram of actions for won games
games_wins_flat = np.array(games_wins).flatten()

actions_won_games = [actions_per_game[i] for i in range(len(actions_per_game)) if games_wins_flat[i] == 1]
actions_lost_games = [actions_per_game[i] for i in range(len(actions_per_game)) if games_wins_flat[i] == 0]

# # Histogram of actions for won games
# plt.hist(actions_won_games, bins=50, alpha=0.5, label='Won Games')

# # Histogram of actions for lost games
# plt.hist(actions_lost_games, bins=50, alpha=0.5, label='Lost Games')

# plt.xlabel('Actions per Game')
# plt.ylabel('Frequency')
# plt.legend()
# plt.title('Number of Actions per Game for Won and Lost Games')
# plt.show()


# Define your 2x2 main subplots
fig = plt.figure(figsize=(15, 15))
gs = gridspec.GridSpec(2, 2, figure=fig)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])

# Define a 1x3 subplot grid within the fourth main subplot
gs_inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 1])

ax4_1 = fig.add_subplot(gs_inner[0, 0])
ax4_2 = fig.add_subplot(gs_inner[0, 1])
ax4_3 = fig.add_subplot(gs_inner[0, 2])


#### HER
# # Set up a 2x2 grid of subplots
# fig, axs = plt.subplots(2, 2, figsize=(15, 15))

# Plot 1: Win rate over games
ax1.set_xlabel('Number of games', fontsize=12)
ax1.set_ylabel('Win rate [%]', fontsize=12)
ax1.plot(range(1, len(best_data) + 1), [element * 100 for element in best_data], label='Raw win rate',  color='blue')
ax1.plot(range(1, len(temp_data) + 1), [element * 100 for element in temp_data], linewidth=2, label='Smooth win rate',  color='orange')
ax1.legend(loc=4, fontsize=12)

# axs[0, 0].set_xlabel('Number of games', fontsize=12)
# axs[0, 0].set_ylabel('Win rate [%]', fontsize=12)
# axs[0, 0].plot(range(1, len(best_data) + 1), [element * 100 for element in best_data], label='Raw win rate',  color='blue')
# axs[0, 0].plot(range(1, len(temp_data) + 1), [element * 100 for element in temp_data], linewidth=2, label='Smooth win rate',  color='orange')
# axs[0, 0].legend(loc=4, fontsize=12)

# Plot 2: Actions per game over time
ax2.plot(games_played_raw, actions_per_game, label='Actions per game', color='blue')
ax2.plot(games_played_ma, actions_per_game_data, label='Moving Average Actions per game', color='orange')
ax2.set_xlabel('Number of Games Played')
ax2.set_ylabel('Actions per Game')
ax2.set_title('Number of Actions per Game over Time')
ax2.legend()
ax2.grid(False)

# axs[0, 1].plot(games_played_raw, actions_per_game, label='Actions per game', color='blue')
# axs[0, 1].plot(games_played_ma, actions_per_game_data, label='Moving Average Actions per game', color='orange')
# axs[0, 1].set_xlabel('Number of Games Played')
# axs[0, 1].set_ylabel('Actions per Game')
# axs[0, 1].set_title('Number of Actions per Game over Time')
# axs[0, 1].legend()
# axs[0, 1].grid(False)

# Plot 3: Histogram of actions for won games
ax3.hist(actions_won_games, bins=50, alpha=0.5, label='Won Games', color='blue')
ax3.hist(actions_lost_games, bins=50, alpha=0.5, label='Lost Games', color='orange')
ax3.set_xlabel('Actions per Game')
ax3.set_ylabel('Frequency')
ax3.set_title('Number of Actions per Game for Won and Lost Games')
ax3.legend()


# axs[1, 0].hist(actions_won_games, bins=50, alpha=0.5, label='Won Games', color='blue')
# axs[1, 0].hist(actions_lost_games, bins=50, alpha=0.5, label='Lost Games', color='orange')
# axs[1, 0].set_xlabel('Actions per Game')
# axs[1, 0].set_ylabel('Frequency')
# axs[1, 0].set_title('Number of Actions per Game for Won and Lost Games')
# axs[1, 0].legend()


avg_win_rate_BT = np.mean(avg_win_rate, axis=(1,2))
# Compute average win rate for each Discount Factor
avg_win_rate_DF = np.mean(avg_win_rate, axis=(0,2))
# Compute average win rate for each Learning Rate
avg_win_rate_LR = np.mean(avg_win_rate, axis=(0,1))
# Plot 4:
# Plot 4.1: Average win rates for each Boltzmann Temperature
bars = ax4_1.bar(boltzmann_temperature, avg_win_rate_BT, width=0.01, align='center', color='orange')
bars[np.argmax(avg_win_rate_BT)].set_color('blue')  # set the color of the highest bar to blue

#ax4_1.bar(boltzmann_temperature, avg_win_rate_BT,width=0.01, align='center', color='orange')
ax4_1.set_xlabel('Boltzmann Temperature')
ax4_1.set_ylabel('Average Win Rate')

# Plot 4.2: Average win rates for each Discount Factor
bars = ax4_2.bar(discount_factor, avg_win_rate_DF,  width=0.01, align='center', color='orange')
bars[np.argmax(avg_win_rate_DF)].set_color('blue')  # set the color of the highest bar to blue

ax4_2.set_xlabel('Discount Factor')
#ax4_2.set_ylabel('Average Win Rate')

# Plot 4.3: Average win rates for each Learning Rate
bars = ax4_3.bar(learning_rate, avg_win_rate_LR,  width=0.01, align='center', color='orange')
bars[np.argmax(avg_win_rate_LR)].set_color('blue')  # set the color of the highest bar to blue

ax4_3.set_xlabel('Learning Rate')
#ax4_3.set_ylabel('Average Win Rate')


# # # Plot 4: Average win rates for each parameter
# # Compute average win rate for each Boltzmann Temperature
# avg_win_rate_BT = np.mean(avg_win_rate, axis=(1,2))
# # Compute average win rate for each Discount Factor
# avg_win_rate_DF = np.mean(avg_win_rate, axis=(0,2))
# # Compute average win rate for each Learning Rate
# avg_win_rate_LR = np.mean(avg_win_rate, axis=(0,1))

# axs[1, 1].bar(boltzmann_temperature, avg_win_rate_BT, alpha=0.5, label='Boltzmann Temperature')
# axs[1, 1].bar(discount_factor, avg_win_rate_DF, alpha=0.5, label='Discount Factor')
# axs[1, 1].bar(learning_rate, avg_win_rate_LR, alpha=0.5, label='Learning Rate')
# axs[1, 1].set_xlabel('Parameter Value')
# axs[1, 1].set_ylabel('Average Win Rate')
# axs[1, 1].set_title('Average Win Rate for Each Parameter')
# axs[1, 1].legend()

# # Placeholder for Plot 4 (If you have another plot)
# # axs[1, 1].plot(...)
# # Compute average win rate for each Boltzmann Temperature
# avg_win_rate_BT = np.mean(avg_win_rate, axis=(1,2))
# # Compute average win rate for each Discount Factor
# avg_win_rate_DF = np.mean(avg_win_rate, axis=(0,2))
# # Compute average win rate for each Learning Rate
# avg_win_rate_LR = np.mean(avg_win_rate, axis=(0,1))
# # Create subplots
# fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# # Plot average win rates for each Boltzmann Temperature
# axs[0].bar(boltzmann_temperature, avg_win_rate_BT)
# axs[0].set_xlabel('Boltzmann Temperature')
# axs[0].set_ylabel('Average Win Rate')

# # Plot average win rates for each Discount Factor
# axs[1].bar(discount_factor, avg_win_rate_DF)
# axs[1].set_xlabel('Discount Factor')
# axs[1].set_ylabel('Average Win Rate')

# # Plot average win rates for each Learning Rate
# axs[2].bar(learning_rate, avg_win_rate_LR)
# axs[2].set_xlabel('Learning Rate')
# axs[2].set_ylabel('Average Win Rate')

# plt.tight_layout()
# plt.show()



plt.tight_layout()
plt.savefig('/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project/src/images/Best_winrate2.png', bbox_inches='tight')
plt.show()