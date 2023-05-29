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
import itertools
import csv
import cv2

device = "/gpu:0" if tf.config.list_physical_devices('GPU') else "/cpu:0"
print(f"Running on {device}")
runMultipleParameters = True
training_started = False

def show_progress(label, full, prog):
    sys.stdout.write("\r{0}: {1}%  [{2}{3}]".format(label, prog, "█"*full, " "*(30-full)))
    sys.stdout.flush()

def plot_heatMap(q):
    state_labels = ["START_AREA", "GOAL_AREA", "WINNING_AREA", "DANGER_AREA", "SAFE_AREA", "DEFAULT_AREA"]
    action_labels = ["STARTING_ACTION", "DEFAULT_ACTION", "INSIDE_GOAL_AREA_ACTION", "ENTER_GOAL_AREA_ACTION", "ENTER_WINNING_AREA_ACTION", "STAR_ACTION", "MOVE_INSIDE_SAFETY_ACTION", "MOVE_OUTSIDE_SAFETY_ACTION", "KILL_PLAYER_ACTION", "DIE_ACTION", "NO_ACTION"]

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


def playGame(q, q_player, training=True, current_game=0, after=0):
    g = ludopy.Game()
    stop_while = False
    if training:
        global training_started
        if not training_started and current_game > after:
            q.training = 1
            training_started = True
        elif training_started:
            q.training = 1
        else:
            q.training = 0
    elif not training:
        q.training = 1 if training and current_game > after else 0
    win_rate = 0

    while not stop_while:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = g.get_observation()
        if player_i == q_player:
            piece_to_move = q.updateQLogic(player_pieces, enemy_pieces, dice, g, there_is_a_winner) if current_game > after else (np.random.choice(move_pieces) if len(move_pieces) > 0 else -1)
            if there_is_a_winner == 1:
                stop_while = True
                win_rate = 1 if player_is_a_winner else 0
        else:
            if len(move_pieces):
                piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
            else:
                piece_to_move = -1

        _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)
        
    # # Save history of the game
    # g.save_hist("Runtime_history.npy")
    # g.save_hist_video("Runtime_history.mp4")


    return g.first_winner_was, q.sum_of_rewards, win_rate


def training(q, number_of_runs_for_training, q_player, after=0):
    array_of_sum_of_rewards = []
    win_rate_list = [0]*after
    #win_rate_list = []
    average_train_win_rates = []  # List to hold average win rates
    for i in range(number_of_runs_for_training):
        first_winner, sum_of_rewards, win_rate = playGame(q, q_player, training=True, current_game=i, after=after)
        array_of_sum_of_rewards.append(sum_of_rewards)
        win_rate_list.append(win_rate)
        q.reset()
        
        # # Compute average win rate after every 45 games
        # if (i+1) % 45 == 0:
        #     average_win_rate = sum(win_rate_list[-45:]) / 45
        #     average_train_win_rates.append(average_win_rate)
        
        # Compute average win rate after every 75 games
        if (i+1) % 50 == 0:
            average_win_rate = sum(win_rate_list[-75:]) / 75
            average_train_win_rates.append(average_win_rate)

        
        if runMultipleParameters == False:
            progress = int(((i + 1) / number_of_runs_for_training) * 30)
            show_progress("training",progress, int(((i + 1) / number_of_runs_for_training) * 100))

    return array_of_sum_of_rewards, win_rate_list, average_train_win_rates

def validation(q, number_of_runs_for_validation, q_player, after=0):
    wins = [0, 0, 0, 0]
    q.training = 0
    array_of_sum_of_rewards = []
    win_rate_list = []
    average_validate_win_rates = []  # List to hold average win rates
    for i in range(number_of_runs_for_validation):
        first_winner, sum_of_rewards, win_rate = playGame(q, q_player, training=False, current_game = i + after, after=after)
        array_of_sum_of_rewards.append(sum_of_rewards)
        win_rate_list.append(win_rate)
        q.reset()
        wins[first_winner] = wins[first_winner] + 1
        
        # # Compute average win rate after every 45 games
        # if (i+1) % 45 == 0:
        #     average_win_rate = sum(win_rate_list[-45:]) / 45
        #     average_validate_win_rates.append(average_win_rate)
        
        # # Compute average win rate after every 60 games
        # if (i+1) % 60 == 0:
        #     average_win_rate = sum(win_rate_list[-50:]) / 50
        #     average_validate_win_rates.append(average_win_rate)

        # # Compute average win rate after every 60 games
        # if (i+1) % 60 == 0:
        #     average_win_rate = sum(win_rate_list[-50:]) / 50
        #     average_validate_win_rates.append(average_win_rate)


        if runMultipleParameters == False:
            progress = int(((i + 1) / number_of_runs_for_validation) * 30)
            show_progress("validation", progress, int(((i + 1) / number_of_runs_for_validation) * 100))
        
    #print("\n")
    return wins, array_of_sum_of_rewards, win_rate_list, average_validate_win_rates



def run():
    # Parameters
    
    # learning_rate = [0.325] # 0.1
    # discount_factor = [0.225] #0.4
    # boltzmann_temperature = [0.175] #0.2 
    
    ####################### BEST #######################
    # learning_rate = [0.325] 
    # discount_factor = [0.175] 
    # boltzmann_temperature = [0.125] 
    
    # learning_rate = [0.25, 0.275 , 0.3, 0.325, 0.35] # 0.3
    # discount_factor = [0.15, 0.175 , 0.2, 0.225, 0.25] # 0.2
    # boltzmann_temperature = [0.15, 0.175 , 0.2, 0.225, 0.25] # 0.2
    
    # learning_rate = [0.3, 0.4, 0.5, 0.6] # 0.1
    # discount_factor = [0.2, 0.3, 0.4, 0.5] #0.4
    # boltzmann_temperature = [0.2, 0.3, 0.4, 0.5] #0.2 
    
    learning_rate = [0.275, 0.300, 0.325, 0.350, 0.375]
    discount_factor = [0.175, 0.200, 0.225, 0.250, 0.275]
    boltzmann_temperature = [0.125, 0.150, 0.175, 0.200, 0.225]

    
    # learning_rate = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    # discount_factor = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    # boltzmann_temperature = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]

    
    # after = 250
    # number_of_runs_for_training = 5000
    # number_of_runs_for_validation = 500
    
    # after = 10
    # number_of_runs_for_training = 6000 # 3600
    # number_of_runs_for_validation = 4000 # 900
    # q_player = 0
    
    # after = 1
    # number_of_runs_for_training = 2000 # 3600
    # number_of_runs_for_validation = 1 # 900
    # q_player = 0
    
    after = 1
    number_of_runs_for_training = 2000 # 7500
    number_of_runs_for_validation = 1 # 900
    q_player = 0
    
    total_iterations = len(boltzmann_temperature) * len(discount_factor) * len(learning_rate) * (number_of_runs_for_training + number_of_runs_for_validation)
    current_iteration = 0
    #print("Total iterations: ", total_iterations)


    # Set for traning
    size_of_win_rate_vec = (len(boltzmann_temperature), len(discount_factor), len(learning_rate), number_of_runs_for_training + number_of_runs_for_validation + after) 
    #size_of_win_rate_vec = (len(boltzmann_temperature), len(discount_factor), len(learning_rate), number_of_runs_for_training + number_of_runs_for_validation) 
    win_rate_vec = np.zeros(size_of_win_rate_vec)
    games_wins = np.zeros(size_of_win_rate_vec)

    for (BTidx, BTval), (DFidx, DFval), (LRidx, LRval) in itertools.product(enumerate(boltzmann_temperature), enumerate(discount_factor), enumerate(learning_rate)):
        if runMultipleParameters == True:
            overall_progress = int((current_iteration / total_iterations) * 30)
            show_progress("Overall progress", overall_progress, int((current_iteration / total_iterations) * 100))

        q = Qlearn.QLearn(q_player)
        q.training = 1
        
        actions_per_game = q.actions_per_game

        q.learning_rate = LRval
        q.discount_factor = DFval
        q.boltzmann_temperature = BTval
        
        global training_started
        training_started = False
        
        array_of_sum_of_rewards, win_rate_list, average_train_win_rates = training(q, number_of_runs_for_training, q_player, after=after)
        current_iteration += number_of_runs_for_training
        #print("Win_rate_list: ",win_rate_list)
        
        wins, array_of_sum_of_rewards_validation, win_rate_list_validation, average_validate_win_rates = validation(q, number_of_runs_for_validation, q_player, after=after)
        current_iteration += number_of_runs_for_validation
        #print("win_rate_list_validation: ", win_rate_list_validation)

        
        #win_rate = (wins[q_player] / number_of_runs_for_validation)        
        win_rate_vec[BTidx][DFidx][LRidx] = win_rate_list + win_rate_list_validation
        results = win_rate_vec[BTidx][DFidx][LRidx]
        win_rate_vec[BTidx][DFidx][LRidx] = np.cumsum(results) / (np.arange(len(results)) + 1)
        #print("win_rate_vec[BTidx][DFidx][LRidx]: ", win_rate_vec[BTidx][DFidx][LRidx])
    
        games_wins[BTidx][DFidx][LRidx] = win_rate_list + win_rate_list_validation
        # Test progress
        #plt.plot(range(len(array_of_sum_of_rewards)),array_of_sum_of_rewards)
        #plot_heatMap(q)
        

        # plot_rewards(array_of_sum_of_rewards, "Training Phase Cumulative Rewards")
        # plot_rewards(array_of_sum_of_rewards_validation, "Validation Phase Cumulative Rewards")

        q.save_QTable("Best_learning_parameters" + str(number_of_runs_for_training) + ".npy")
        
        
        average_win_rates = average_train_win_rates + average_validate_win_rates

        with open('my_win_rates_new.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["episode", "my_win_rate"])
            for i, rate in enumerate(average_win_rates):
                writer.writerow([i+1, rate])
                
    # Save data and parameters
    save_data_and_parameters(win_rate_vec, boltzmann_temperature, discount_factor, learning_rate, number_of_runs_for_training, number_of_runs_for_validation, actions_per_game, games_wins, average_win_rates)
    return True

def save_data_and_parameters(win_rate_vec, explore_rate_vec, discount_factor_vec, learning_rate_vec, number_of_runs_for_training, number_of_runs_for_validation, actions_per_game, games_wins, average_win_rates):
    # specify the folder path
    folder_path = os.path.join(os.getcwd(), "/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project/src/data")

    # create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # save the data file to the folder
    data_file_path = os.path.join(folder_path, "data.npy")
    np.save(data_file_path, [win_rate_vec, actions_per_game, games_wins, average_win_rates])

    # save the parameters file to the folder
    param_file_path = os.path.join(folder_path, "parameters.npy")
    np.save(param_file_path, [explore_rate_vec, discount_factor_vec, learning_rate_vec, number_of_runs_for_training, number_of_runs_for_validation])

class MyTestCase(unittest.TestCase):
    def test_something(self):
        with tf.device('/device:GPU:0'):
            self.assertEqual(True, run())


if __name__ == '__main__':
    unittest.main()

