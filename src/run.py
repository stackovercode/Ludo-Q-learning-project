import sys
sys.path.insert(0, "/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project")
import numpy as np
import Qlearn
import ludopy 
import unittest
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import time

device = "/gpu:0" if tf.config.list_physical_devices('GPU') else "/cpu:0"
print(f"Running on {device}")

def plot_heatMap(q):
    state_labels = ["start", "goal", "winning", "danger", "safe", "default"]
    action_labels = ["Starting", "Default", "Inside_goal", "Enter_goal", "Enter_winning", "Star", "Move_safety", "Move_away_safety", "Kill_enemy", "Die_action", "No_action"]

    fig, ax = plt.subplots()
    im = ax.imshow(q.Q_table, cmap='coolwarm')

    ax.set_xticks(np.arange(len(action_labels)))
    ax.set_yticks(np.arange(len(state_labels)))
    ax.set_xticklabels(action_labels)
    ax.set_yticklabels(state_labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(state_labels)):
        for j in range(len(action_labels)):
            text = ax.text(j, i, int(q.Q_table[i, j] * 100), ha="center", va="center", color="black")

    ax.set_title("Q-table")
    fig.tight_layout()
    plt.show()

def play_game(q, q_player, training=True):
    g = ludopy.Game()
    stop_while = False
    q.training = 1 if training else 0

    while not stop_while:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
        there_is_a_winner), player_i = g.get_observation()

        if player_i == q_player:
            piece_to_move = q.updateQTable(player_pieces, enemy_pieces, dice, g, there_is_a_winner)
            if there_is_a_winner == 1:
                stop_while = True
        else:
            if len(move_pieces):
                piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
            else:
                piece_to_move = -1

        _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)

    return g.first_winner_was, q.sum_of_rewards

def training_phase(q, number_of_runs_for_training, q_player):
    array_of_sum_of_rewards = []
    for k in range(number_of_runs_for_training):
        print('Number of learning games: ', k, ' ER: ', q.explore_rate, ' DF: ', q.discount_factor, ' LR: ', q.learning_rate)
        first_winner, sum_of_rewards = play_game(q, q_player, training=True)
        array_of_sum_of_rewards.append(sum_of_rewards)
        q.reset()

    return array_of_sum_of_rewards

def validation_phase(q, number_of_runs_for_validation, q_player):
    wins = [0, 0, 0, 0]
    q.training = 0
    array_of_sum_of_rewards = []

    for j in range(number_of_runs_for_validation):
        first_winner, sum_of_rewards = play_game(q, q_player, training=False)
        array_of_sum_of_rewards.append(sum_of_rewards)
        q.reset()
        wins[first_winner] = wins[first_winner] + 1

    return wins, array_of_sum_of_rewards

def run():
    # Parameters
 #  Explore rate: 0.05, discount rate: 0.4 and learning rate: 0.1
    learning_rate_vec = [0.6] # 0.1
    discount_factor_vec = [0.4] #0.4
    explore_rate_vec = [0.4] #0.05
    # learning_rate_vec = [0.1, 0.2, 0.3, 0.4, 0.5]
    # discount_factor_vec = [0.1, 0.2, 0.3, 0.4, 0.5]
    # explore_rate_vec = [0.05, 0.10, 0.15, 0.2]
    after = 0 
    number_of_runs_for_training = 800
    number_of_runs_for_validation = 200
    q_player = 0

    size_of_win_rate_vec = (len(explore_rate_vec), len(discount_factor_vec), len(learning_rate_vec), number_of_runs_for_training)
    win_rate_vec = np.zeros(size_of_win_rate_vec)

    for ER_index, ER_value in enumerate(explore_rate_vec):
        for DF_index, DF_value in enumerate(discount_factor_vec):
            for LR_index, LR_value in enumerate(learning_rate_vec):
                q = Qlearn.QLearn(q_player)
                q.training = 1

                q.learning_rate = LR_value
                q.discount_factor = DF_value
                q.explore_rate = ER_value

                array_of_sum_of_rewards = training_phase(q, number_of_runs_for_training, q_player)
                wins, array_of_sum_of_rewards_validation = validation_phase(q, number_of_runs_for_validation, q_player)
                

                win_rate = (wins[q_player] / number_of_runs_for_validation)
                print('Win rate: ', win_rate)
                win_rate_vec[ER_index][DF_index][LR_index] = win_rate

                # Test progress
                plt.plot(range(len(array_of_sum_of_rewards)),array_of_sum_of_rewards)
                plot_heatMap(q)

                q.save_QTable("Best_learning_parameters" + str(number_of_runs_for_training) + ".npy")

    # Save data and parameters
    #save_data_and_parameters(win_rate_vec, explore_rate_vec, discount_factor_vec, learning_rate_vec, number_of_runs_for_training, number_of_runs_for_validation)
    # specify the folder path
    folder_path = os.path.join(os.getcwd(), "/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project/src/data")
    test_name = ""

    # create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # save the data file to the folder
    data_file_path = os.path.join(folder_path, test_name + "data.npy")
    np.save(data_file_path, win_rate_vec)

    # save the parameters file to the folder
    param_file_path = os.path.join(folder_path, test_name + "parameters.npy")
    np.save(param_file_path, [explore_rate_vec, discount_factor_vec, learning_rate_vec, number_of_runs_for_training, number_of_runs_for_validation])

    
    
    return True


# def save_data_and_parameters(win_rate_vec, explore_rate_vec, discount_factor_vec, learning_rate_vec, number_of_runs_for_training, number_of_runs_for_validation):
#     # specify the folder path
#     folder_path = os.path.join(os.getcwd(), "/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project/src/data")

#     # create the folder if it doesn't exist
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)

#     # save the data file to the folder
#     data_file_path = os.path.join(folder_path, "data.npy")
#     np.save(data_file_path, win_rate_vec)

#     # save the parameters file to the folder
#     param_file_path = os.path.join(folder_path, "parameters.npy")
#     np.save(param_file_path, [explore_rate_vec, discount_factor_vec, learning_rate_vec, number_of_runs_for_training, number_of_runs_for_validation])

class MyTestCase(unittest.TestCase):
    def test_something(self):
        with tf.device('/device:GPU:0'):
            self.assertEqual(True, run())


if __name__ == '__main__':
    unittest.main()

