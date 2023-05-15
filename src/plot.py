import sys
import numpy as np
import matplotlib.pyplot as plt

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
games_wins = data[2]

explore_rate_vec = parameters[0]
discount_factor_vec = parameters[1]
learning_rate_vec = parameters[2]

highest_win_rate = 0
best_index = [0,0,0]
best_data = []
temp_data = []

for ER_index, ER_value in enumerate(explore_rate_vec):
    for DF_index, DF_value in enumerate(discount_factor_vec):
        for LR_index, LR_value in enumerate(learning_rate_vec):
            print(win_rate_vec[ER_index][DF_index][LR_index])  # Debug print
            temp_win_rate = np.sum(win_rate_vec[ER_index][DF_index][LR_index]) / len(win_rate_vec[ER_index][DF_index][LR_index])
            if temp_win_rate > highest_win_rate :
                highest_win_rate = temp_win_rate
                temp_data = exponential_moving_average(win_rate_vec[ER_index][DF_index][LR_index], 0.5)
                best_data = win_rate_vec[ER_index][DF_index][LR_index]
                best_index = [ER_value, DF_value, LR_value]

print(highest_win_rate, best_index)
fig, axs = plt.subplots()
axs.set_xlabel('Number of games', fontsize=12)
axs.set_ylabel('Win rate [%]', fontsize=12)
axs.plot(range(1, len(best_data) + 1), [element * 100 for element in best_data], label = 'Raw win rate')
axs.plot(range(1, len(temp_data) + 1), [element * 100 for element in temp_data], linewidth = 3, label = 'Smooth win rate')
axs.legend(loc=4, fontsize=12)

plt.savefig('/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project/src/images/Best_winrate2.png', bbox_inches='tight')

plt.show()

games_played_raw = list(range(1, len(actions_per_game) + 1))

# Generate x values for moving average data
actions_per_game_data = moving_average(actions_per_game, 15)
games_played_ma = list(range(1, len(actions_per_game_data) + 1))  

plt.figure(figsize=(10, 6))

# Plot raw data
plt.plot(games_played_raw, actions_per_game, label='Actions per game')

# Plot moving average data
plt.plot(games_played_ma, actions_per_game_data, label='Moving Average Actions per game')

plt.xlabel('Number of Games Played')
plt.ylabel('Actions per Game')
plt.title('Number of Actions per Game over Time')
plt.legend()
plt.grid(False)
plt.show()

# Histogram of actions for won games
games_wins_flat = np.array(games_wins).flatten()

actions_won_games = [actions_per_game[i] for i in range(len(actions_per_game)) if games_wins_flat[i] == 1]
actions_lost_games = [actions_per_game[i] for i in range(len(actions_per_game)) if games_wins_flat[i] == 0]

# Histogram of actions for won games
plt.hist(actions_won_games, bins=50, alpha=0.5, label='Won Games')

# Histogram of actions for lost games
plt.hist(actions_lost_games, bins=50, alpha=0.5, label='Lost Games')

plt.xlabel('Actions per Game')
plt.ylabel('Frequency')
plt.legend()
plt.title('Number of Actions per Game for Won and Lost Games')
plt.show()
