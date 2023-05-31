import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def exponential_average(data, alpha):
    ema = [data[0]]  # starts from the first data point
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[i-1])
    return ema

# Reading CSV
my_win_rates_df = pd.read_csv('my_win_rates.csv')
other_win_rates_df = pd.read_csv('other_win_rates.csv')

# Calculate rolling averages
my_win_rates_data = exponential_average(my_win_rates_df['my_win_rate'], 0.1)  # replace 'my_win_rate' with your actual column name
other_win_rates_data =  exponential_average(other_win_rates_df['other_win_rate'], 0.1)  # replace 'other_win_rate' with your actual column name

# Create a list of episode numbers matching the length of your win rates
episodes = list(range(1, len(my_win_rates_data) + 1))

# Create a scatter plot

plt.scatter(episodes, exponential_average(my_win_rates_df['my_win_rate'], 0.3), color='lightblue', linewidth=1.0)
plt.plot(episodes,  my_win_rates_data, label='Own method', color='orange', linewidth=2.0)

plt.scatter(episodes, exponential_average(other_win_rates_df['other_win_rate'], 0.3), color='lightblue', linewidth=1.0)
plt.plot(episodes,  other_win_rates_data, label='Comp. method', color='red', linewidth=2.0)

plt.legend(loc=4, prop={'size': 16})  # Change font size of legend

# set the title and labels with increased font size
plt.title('Average Win Rates over Episodes', fontsize=16)
plt.xlabel('Episode', fontsize=14)
plt.ylabel('Average Win Rate', fontsize=14)

# Set the y-axis limits and step size
plt.ylim([0, 0.9])
plt.yticks([i/10 for i in range(0,10)])

# Increase font size of ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

#plt.savefig('/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project/src/images/compare_averge_winrate.png', bbox_inches='tight')
plt.savefig('/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project/src/images/Plot_1_Best_winrate.svg', format='svg', dpi=1200)

plt.show()