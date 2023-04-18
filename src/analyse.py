import numpy as np
import matplotlib.pyplot as plt

def moving_average(list,N):
	# https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
	cumsum, moving_aves = [0], []
	for i, x in enumerate(list, 1):
		cumsum.append(cumsum[i - 1] + x)
		if i >= N:
			moving_ave = (cumsum[i] - cumsum[i - N]) / N
			moving_aves.append(moving_ave)
	return moving_aves



file_name = 'Test1'


parameters_1 = np.load('/home/reventlov/TAI/Project/Ludo Q-learning project/src/data/Test_run_parameters.npy',allow_pickle=True)
win_rate_vec = np.load('/home/reventlov/TAI/Project/Ludo Q-learning project/src/data/Test_run_data.npy')

explore_rate_vec = parameters_1[0]

discount_factor_vec = parameters_1[1]

learning_rate_vec = parameters_1[2]

highest_win_rate = 0
best_index = [0,0,0]
best_data = []

for ER_index, ER_value in enumerate(explore_rate_vec):
	for DF_index, DF_value in enumerate(discount_factor_vec):
		for LR_index, LR_value in enumerate(learning_rate_vec):
			temp_win_rate = np.sum(win_rate_vec[ER_index][DF_index][LR_index][20:])/len(win_rate_vec[ER_index][DF_index][LR_index][20:]) # 700 / 700
			if temp_win_rate > highest_win_rate :
				highest_win_rate = temp_win_rate
				temp_data = moving_average(win_rate_vec[ER_index][DF_index][LR_index],15)
				best_data = win_rate_vec[ER_index][DF_index][LR_index]
				best_index = [ER_value,DF_value,LR_value]

print(highest_win_rate,best_index)
temp_data= moving_average(temp_data,5)
fig, axs = plt.subplots()
axs.set_xlabel('Number of games', fontsize=12)
axs.set_ylabel('Win rate [%]', fontsize=12)
axs.plot(range(1,len(best_data)+1), [element * 100 for element in best_data], label = 'Unfiltered')
axs.plot(range(1,len(temp_data)+1), [element * 100 for element in temp_data],linewidth = 3,label = 'Filtered')
axs.legend(loc=4,fontsize=12)

plt.savefig('/home/reventlov/TAI/Project/Ludo Q-learning project/src/images/Best_winrate.png',bbox_inches='tight')

plt.show()

