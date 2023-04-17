import unittest
import cv2
import sys
import os
import random

# Add the path to the LUDOpy directory to the Python path
# Add the path to the parent directory of LUDOpy to the Python path
ludopy_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'LUDOpy'))
parent_dir = os.path.dirname(ludopy_dir)
sys.path.append(parent_dir)

# Import the player module from the LUDOpy package
from LUDOpy.ludopy import player

import numpy as np
import matplotlib.pyplot as plt


# Number of game bricks used by every player
no_gameBricks = 4
number_States = 6
number_Actions = 11

## Chosen states for the game areas: startArea, goalArea, winningArea, dangerArea, safeArea, defaultArea
startArea = 0
goalArea = 1
winningArea = 2
dangerArea = 3
safeArea = 4
defaultArea = 5

starting_action = 0
default_action = 1
inside_goalArea_action = 2
enter_goalArea_action = 3
enter_winningArea_action = 4
star_action = 5
move_inside_safety_action = 6
move_outside_safety_action = 7
kill_player_action = 8
die_action = 9
no_action = 10

def plot_heat_map(q):
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

def count(list, value):
    no_Ocurrences = 0;
    for element in list:
        if element == value:
            no_Ocurrences = no_Ocurrences + 1
    return no_Ocurrences


class QLearning:
    def __init__(self, index):
        # Parameters for the algorithm
        self.learning_rate = 0.7 # alpha
        self.discount_factor = 0.4 #gama pre 0.01
        self.explore_rate = 0.4 # epsilon
        self.sum_of_rewards = 0.0
        self.training = 1 # determineds if the q table is updated and if there is going to be any explorations.
        self.Q_table = np.zeros((number_States, number_Actions))

        #Parameters for the interpretations of the game
        self.player_index = index
        self.current_state = [startArea] * 4
        self.last_action = no_action
        self.last_player_pieces = [0] * 4
        self.number_of_wins = 0
        self.number_of_games = 0
        self.last_state = startArea

    def reset_game(self):
        self.current_state = [startArea] * 4
        self.last_action = None
        self.last_state = startArea
        self.last_player_pieces = [0] * 4
        self.sum_of_rewards = 0.0
        self.number_of_games += 1

     # State determination logic
    def determined_state(self,player_pieces, enemy_pieces, game):
        state_of_pieces = [startArea, startArea, startArea, startArea]
        for piece_index in range(no_gameBricks):

            if player_pieces[piece_index] == player.HOME_INDEX:  # home
                state_of_pieces[piece_index] = startArea
            elif player_pieces[piece_index] in player.HOME_AREAL_INDEXS:  # goal zone
                state_of_pieces[piece_index] = goalArea
            elif player_pieces[piece_index] == player.GOAL_INDEX:  # goal
                state_of_pieces[piece_index] = winningArea
            elif (player_pieces[piece_index] in player.GLOB_INDEXS) or (
                    player_pieces[piece_index] == player.START_INDEX or count(player_pieces,player_pieces[piece_index])>1 ):
                state_of_pieces[piece_index] = safeArea
            else:
                state_determined = 0
                for index in range(len(player.LIST_ENEMY_GLOB_INDEX)):
                    if player_pieces[piece_index] == player.LIST_ENEMY_GLOB_INDEX[index]:
                        if player.HOME_INDEX in enemy_pieces[index]:
                            if not (index in game.ghost_players):
                                state_of_pieces[piece_index] = dangerArea
                            else:
                                state_of_pieces[piece_index] = safeArea
                            state_determined = 1
                            break
                if state_determined == 0:
                    if player_pieces[piece_index] in player.STAR_INDEXS:
                        if player_pieces[piece_index] in player.STAR_INDEXS[1::2]:
                            range_to_look_for_enemies = list(range(1, 7))
                            range_to_look_for_enemies.extend(list(range(8, 14)))
                        else:
                            range_to_look_for_enemies = list(range(1, 13))
                    else:
                        range_to_look_for_enemies = list(range(1, 7))
                    piece_pos = player_pieces[piece_index]
                    for index in range_to_look_for_enemies:
                        piece_pos = player_pieces[piece_index] - index
                        if piece_pos < 1:
                            piece_pos = 52 + piece_pos
                        enemy_at_pos, _ = player.get_enemy_at_pos(piece_pos, enemy_pieces)
                        if not (enemy_at_pos == player.NO_ENEMY):
                            state_of_pieces[piece_index] = dangerArea
                            state_determined = 1
                            break
                if state_determined == 0:
                    state_of_pieces[piece_index] = defaultArea
        return state_of_pieces

    # Possible action determination logic
    def determined_possible_actions(self, player_pieces, enemy_pieces, dice):
        possible_actions = [default_action, default_action, default_action, default_action]

        for piece_index in range(no_gameBricks):
            old_piece_pos = player_pieces[piece_index]
            new_piece_pos = old_piece_pos + dice
            enemy_at_pos, enemy_pieces_at_pos = player.get_enemy_at_pos(new_piece_pos, enemy_pieces)
            if old_piece_pos == player.GOAL_INDEX or (old_piece_pos == player.HOME_INDEX and dice < 6):  # piece at goal
                possible_actions[piece_index] = no_action

            elif old_piece_pos == player.HOME_INDEX and dice == 6:  # move out of home
                possible_actions[piece_index] = starting_action

            elif new_piece_pos in player.STAR_INDEXS:  # use a star to jump
                possible_actions[piece_index] = star_action

            elif new_piece_pos == player.GOAL_INDEX or new_piece_pos == player.STAR_AT_GOAL_AREAL_INDX:  # enter goal
                possible_actions[piece_index] = enter_winningArea_action

            elif new_piece_pos in player.GLOB_INDEXS:  # globe not owned
                if enemy_at_pos != player.NO_ENEMY:
                    possible_actions[piece_index] = die_action
                else:
                    possible_actions[piece_index] = move_inside_safety_action
            elif count(player_pieces,new_piece_pos) > 0:
                possible_actions[piece_index] = move_inside_safety_action
            elif new_piece_pos in player.LIST_ENEMY_GLOB_INDEX:
                # Get the enemy their own the glob
                globs_enemy = player.LIST_TAILE_ENEMY_GLOBS.index(player.BORD_TILES[new_piece_pos])
                # Check if there is an enemy at the glob
                if enemy_at_pos != player.NO_ENEMY:
                    # If there is another enemy then send them home and move there
                    if enemy_at_pos != globs_enemy:
                        possible_actions[piece_index] = kill_player_action
                    # If it is the same enemy that is there then move there
                    else:
                        possible_actions[piece_index] = die_action
                # If there are not any enemys at the glob then move there
                else:
                    possible_actions[piece_index] = move_inside_safety_action

            elif enemy_at_pos != player.NO_ENEMY:
                if len(enemy_pieces_at_pos) == 1:
                    possible_actions[piece_index] = kill_player_action
                else:
                    possible_actions[piece_index] = die_action

            elif old_piece_pos < 53 and new_piece_pos > 52:
                possible_actions[piece_index] = enter_goalArea_action

            elif old_piece_pos in player.GLOB_INDEXS or count(player_pieces, old_piece_pos) > 1:
                possible_actions[piece_index] = move_outside_safety_action

            elif new_piece_pos > 52 and not (new_piece_pos == player.GOAL_INDEX):
                possible_actions[piece_index] = inside_goalArea_action
            else:
                possible_actions[piece_index] = default_action
        return possible_actions

    # Reward calculation logic
    def get_reward(self, player_pieces, current_states, there_is_a_winner):
        reward = 0.0

        if self.last_action == starting_action:
            reward += 0.3
        if self.last_action == kill_player_action:
            reward += 0.2
        if self.last_action == die_action:
            reward += -0.8
        if self.last_action == default_action:
            reward += 0.05
        if self.last_action == inside_goalArea_action:
            reward += 0.05
        if self.last_action == enter_goalArea_action:
            reward += 0.2
        if self.last_action == star_action:
            reward += 0.15
        if self.last_action == enter_winningArea_action:
            reward += 0.25
        if self.last_action == move_outside_safety_action:
            reward += -0.1
        if self.last_action == move_inside_safety_action:
            reward += 0.1
        if self.last_action == no_action:
            reward += -0.1

        for i in range(no_gameBricks):
            if self.last_player_pieces[i] > 0 and player_pieces[i] == 0: 
                # A piece has been moved home
                reward += -0.25
                break

        return reward

    # Action selection logic
    def pick_action(self,piece_states,piece_actions):
        best_action_player = -1
        if not (piece_actions.count(no_action) == len(piece_actions)):
            if self.explore_rate == 0 or float(random.randint(1, 100))/100 >self.explore_rate:
                max_q = -1000
                best_action_player = -1
                for i in range(4):
                    if not(piece_actions[i] == no_action):
                        if max_q < self.Q_table[piece_states[i]][piece_actions[i]]:
                            max_q = self.Q_table[piece_states[i]][piece_actions[i]]
                            best_action_player = i
            else:
                while True:
                    best_action_player = random.randint(0, 3)
                    if not(piece_actions[best_action_player] == no_action):
                        break
        return best_action_player


    # Update Q-table logic
    def update_q_table(self, player_pieces, enemy_pieces, dice, game, there_is_a_winner):
        current_actions = self.determined_possible_actions( player_pieces,enemy_pieces,dice)
        current_states = self.determined_state(player_pieces, enemy_pieces, game)
        piece_index = self.pick_action(current_states, current_actions)
        if self.training == 1 and not(piece_index == -1):

            reward = self.get_reward(player_pieces, current_states, there_is_a_winner)

            self.sum_of_rewards += reward
            current_q_value = self.Q_table[current_states[piece_index]][current_actions[piece_index]]
            last_q_value = self.Q_table[self.last_state][self.last_action]
            self.Q_table[self.last_state][self.last_action] += \
                self.learning_rate*(reward+self.discount_factor*current_q_value-last_q_value)
            #print("REWARD",reward)
            #print("SUM REWARD", self.sum_of_rewards)
            #print("Q_TABLE",self.Q_table)
            self.last_player_pieces = player_pieces
            self.last_state = current_states[piece_index]
            self.last_action = current_actions[piece_index]
        return piece_index

    def save_Q_table(self,file_name):
        file_ext = file_name.split(".")[-1]
        assert file_ext == "npy", "The file extension has to be npy (numpy file)"
        np.save(file_name, self.Q_table)

    def load_Q_table(self,file_name):
        file_ext = file_name.split(".")[-1]
        assert file_ext == "npy", "The file extension has to be npy (numpy file)"
        self.Q_table = np.load(file_name)