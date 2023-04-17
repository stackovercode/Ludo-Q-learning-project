import numpy as np
import matplotlib.pyplot as plt

def moving_average(list,N):
	cumsum, moving_aves = [0], []
	for i, x in enumerate(list, 1):
		cumsum.append(cumsum[i - 1] + x)
		if i >= N:
			moving_ave = (cumsum[i] - cumsum[i - N]) / N
			# can do stuff with moving_ave here
			moving_aves.append(moving_ave)
	return moving_aves

file_name = 'Test1'

parameters_1 = np.load('/home/reventlov/TAI/Project/Ludo Q-learning project/Test1_2_parameters.npy', allow_pickle=True)
win_rate_vec_1 = np.load('/home/reventlov/TAI/Project/Ludo Q-learning project/Test1_2_data.npy')

parameters_2 = np.load('/home/reventlov/TAI/Project/Ludo Q-learning project/Test2_2_parameters.npy', allow_pickle=True)
win_rate_vec_2 = np.load('/home/reventlov/TAI/Project/Ludo Q-learning project/Test2_2_data.npy')

explore_rate_vec_1 = parameters_1[0]
explore_rate_vec_2 = parameters_2[0]

discount_factor_vec_1 = parameters_1[1]
discount_factor_vec_2 = parameters_2[1]

learning_rate_vec_1 = parameters_1[2]
learning_rate_vec_2 = parameters_2[2]

highest_win_rate = 0
best_index = [0, 0, 0]
best_data = []

for i in range(2):
	if i == 1:
		explore_rate_vec = explore_rate_vec_1
		discount_factor_vec = discount_factor_vec_1
		learning_rate_vec = learning_rate_vec_1
		win_rate_vec = win_rate_vec_1
	else:
		explore_rate_vec = explore_rate_vec_2
		discount_factor_vec = discount_factor_vec_2
		learning_rate_vec = learning_rate_vec_2
		win_rate_vec = win_rate_vec_2
	for ER_index, ER_value in enumerate(explore_rate_vec):
		for DF_index, DF_value in enumerate(discount_factor_vec):
			for LR_index, LR_value in enumerate(learning_rate_vec):
				temp_win_rate = np.sum(win_rate_vec[ER_index][DF_index][LR_index][300:])/len(win_rate_vec[ER_index][DF_index][LR_index][300:])
				if temp_win_rate > highest_win_rate :
					highest_win_rate = temp_win_rate
					temp_data = moving_average(win_rate_vec[ER_index][DF_index][LR_index], 10)
					best_data = win_rate_vec[ER_index][DF_index][LR_index]
					best_index = [ER_value,DF_value,LR_value]

print(highest_win_rate,best_index)
temp_data= moving_average(temp_data,5)
fig, axs = plt.subplots()
axs.set_xlabel('Number of games')
axs.set_ylabel('Win rate [%]')
axs.plot(range(1,len(best_data)+1), [element * 100 for element in best_data])
axs.plot(range(1,len(temp_data)+1), [element * 100 for element in temp_data],linewidth =4)
plt.savefig('/home/reventlov/TAI/Project/github/ludo_game_AI2-main/Images_report/Best_winrate3.png',bbox_inches='tight')

for i in range(2):
	if i == 1:
		explore_rate_vec = explore_rate_vec_1
		discount_factor_vec = discount_factor_vec_1
		learning_rate_vec = learning_rate_vec_1
		win_rate_vec = win_rate_vec_1
	else:
		explore_rate_vec = explore_rate_vec_2
		discount_factor_vec = discount_factor_vec_2
		learning_rate_vec = learning_rate_vec_2
		win_rate_vec = win_rate_vec_2
	for ER_index, ER_value in enumerate(explore_rate_vec):
		boxplot_vec = []
		label_data = []
		fig, axs = plt.subplots(figsize=(10, 6))
		fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
		axs.set_title('Explore rate ' + str(ER_value), fontsize=16)
		for DF_index, DF_value in enumerate(discount_factor_vec):
			for LR_index, LR_value in enumerate(learning_rate_vec):
				temp_data = win_rate_vec[ER_index][DF_index][LR_index][300:]
				boxplot_vec.append([element * 100 for element in temp_data])
				label_data = label_data +[ '['+str(DF_value) + ',' + str(LR_value)+']' ]
		axs.boxplot(boxplot_vec)
		plt.ylim(0, 110)
		axs.set_xlabel('Learning parameters [DF, LR]', fontsize=16)
		axs.set_ylabel('Win rate [%]', fontsize=16)

		plt.xticks(range(1,len(label_data)+1), label_data, rotation=45, fontsize=12)
		plt.savefig('/home/reventlov/TAI/Project/ludo_game_AI2-main/Images_report/training_ER_'+str(ER_value) + '.png',
					bbox_inches='tight')

for i in range(1,2):
	if i == 1:
		explore_rate_vec = explore_rate_vec_1
		discount_factor_vec = discount_factor_vec_1
		learning_rate_vec = learning_rate_vec_1
		win_rate_vec = win_rate_vec_1
	else:
		explore_rate_vec = explore_rate_vec_2
		discount_factor_vec = discount_factor_vec_2
		learning_rate_vec = learning_rate_vec_2
		win_rate_vec = win_rate_vec_2
	for ER_index, ER_value in enumerate(explore_rate_vec):
		boxplot_vec = []
		label_data = []
		for DF_index, DF_value in enumerate(discount_factor_vec):

			fig, axs = plt.subplots()
			axs.title.set_text('Win Rate ' + str(ER_index))
			for LR_index, LR_value in enumerate(learning_rate_vec):
				temp_data = moving_average(win_rate_vec[ER_index][DF_index][LR_index], 20)
				axs.plot(range(1,len(temp_data)+1), temp_data,label = ' ER: '+ str(ER_value) + ' DF: ' + str(DF_value) + ' LR: ' + str(LR_value))
				axs.legend(loc=4)
				boxplot_vec.append([element * 100 for element in temp_data])
				label_data = label_data +[str(ER_value) + ',' + str(DF_value) + ',' + str(LR_value)]
	# Best set of parameters 63% win rate [0.05, 0.4, 0.1]
plt.show()




