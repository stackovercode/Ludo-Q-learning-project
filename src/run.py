import sys
sys.path.insert(0, "/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project")
import numpy as np
import QLearning
import ludopy 
import unittest
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import multiprocessing as mp
import time

device = "/gpu:0" if tf.config.list_physical_devices('GPU') else "/cpu:0"
print(f"Running on {device}")

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time for {func.__name__}: {end_time - start_time} seconds")
        return result
    return wrapper

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

#@timer
def run():
    #  Explore rate: 0.05, discount rate: 0.4 and learning rate: 0.1
    learning_rate_vec = [0.6] # 0.1
    discount_factor_vec = [0.4] #0.4
    explore_rate_vec = [0.4] #0.05
    # learning_rate_vec = [0.1, 0.2, 0.3, 0.4, 0.5]
    # discount_factor_vec = [0.1, 0.2, 0.3, 0.4, 0.5]
    # explore_rate_vec = [0.05, 0.10, 0.15, 0.2]

    # determines after how many epochs the Q-learning algorithm will stop updating the Q-values and only use the learned Q-values to play the game.
    after = 0 
    
    number_of_runs_for_training = 800 # number of games played for training the Q-values
    number_of_runs_for_validation = 200 # number of games played to test the learned Q-values

    q_player = 0

    size_of_win_rate_vec = (len(explore_rate_vec),len(discount_factor_vec),len(learning_rate_vec), number_of_runs_for_training)
    win_rate_vec = np.zeros(size_of_win_rate_vec)

    for ER_index, ER_value in enumerate(explore_rate_vec):
        for DF_index, DF_value in enumerate(discount_factor_vec):
            for LR_index, LR_value in enumerate(learning_rate_vec):
                q = QLearning.QLearn(q_player)
                q.training = 1

                q.learning_rate = LR_value
                q.discount_factor = DF_value
                q.explore_rate = ER_value

                array_of_sum_of_rewards = []

                for k in range(number_of_runs_for_training):
                    print('Test:   Number of learning games: ', k, ' ER: ', q.explore_rate, ' DF: ', q.discount_factor, ' LR: ', q.learning_rate)
                    g = ludopy.Game()
                    stop_while = False
                    q.training = 1
                    #number_of_wins_W = 0

                    while not stop_while:
                        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
                         there_is_a_winner), player_i = g.get_observation()

                        if player_i == q_player:
                            piece_to_move = q.updateQTable(player_pieces, enemy_pieces, dice, g, there_is_a_winner)
                            if there_is_a_winner == 1:
                                stop_while = True
                                #number_of_wins_W += 1
                        else:
                            if len(move_pieces):
                                piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
                            else:
                                piece_to_move = -1

                        _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)

                    q.reset()
                    if after < k:
                        wins = [0, 0, 0, 0]
                        q.training = 0
                        #number_of_wins_WO = 0

                        number_of_steps = 0
                        for j in range(number_of_runs_for_validation):
                            g = ludopy.Game()
                            stop_while = False
                            while not stop_while:
                                (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
                                 there_is_a_winner), player_i = g.get_observation()
                                if player_i == q_player:
                                    number_of_steps = number_of_steps + 1
                                    piece_to_move = q.updateQTable(player_pieces, enemy_pieces, dice, g, there_is_a_winner)
                                    if there_is_a_winner == 1:
                                        stop_while = True
                                        #number_of_wins_WO += 1
                                else:
                                    if len(move_pieces):
                                        piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
                                    else:
                                        piece_to_move = -1
                                _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)

                            array_of_sum_of_rewards.append(q.sum_of_rewards)

                            # # Test progress
                            # plt.plot(range(len(array_of_sum_of_rewards)),array_of_sum_of_rewards)
                            # plot_heatMap(q)
                            # plt.show()

                            q.reset()
                            wins[g.first_winner_was] = wins[g.first_winner_was] + 1
                        win_rate_vec[ER_index][DF_index][LR_index][k] = (wins[q_player] / number_of_runs_for_validation)
                        print('Win rate: ', wins[q_player] / number_of_runs_for_validation)

                # Test progress
                plt.plot(range(len(array_of_sum_of_rewards)),array_of_sum_of_rewards)
                plot_heatMap(q)
                #plt.show()
                q.save_QTable("Best_learning_parameters" + str(k) + ".npy")

    # specify the folder path
    folder_path = os.path.join(os.getcwd(), "/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project/src/data")
    #test_name = "Test_run"
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


class MyTestCase(unittest.TestCase):
    def test_something(self):
        with tf.device('/device:GPU:0'):
            self.assertEqual(True, run())


if __name__ == '__main__':
    unittest.main()

