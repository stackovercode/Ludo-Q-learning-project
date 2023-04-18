from ludopy import player
import numpy as np
import sys
import random
import matplotlib.pyplot as plt
sys.path.append("../")

number_of_pieces = 4
# states = ["home", "goal_zone", "goal", "danger", "glob","safe"]
total_number_of_states = 60

# actions = ["move_out", "normal", "goal_zone", "goal", "star", "globe", "protect", "kill", "die",  "nothing"]
total_number_of_actions = 11
move_out_action = 0
normal_action = 1
in_goal_zone_action = 2
enter_goal_zone_action = 3
enter_goal_action = 4
use_star_action = 5
move_to_safety_action = 6
move_away_from_safe_action = 7
kill_enemy_action = 8
suicide_action = 9
no_action = 10


def plot_heat_map(q):
    actions = ["Move_out", "Normal", "In_goal_zone",
               "Enter_goal_zone", "enter_goal_action", "Use_star", "Move_to_safety", "Move_away_from_safe",
               "Kill_enemy", "Suicide_action", "no_action"]

    fig, ax = plt.subplots()

    plt.imshow(q.Q_table)

    # We want to show all ticks...
    ax.set_xticks(np.arange(total_number_of_actions))
    ax.set_yticks(np.arange(total_number_of_states))
    # ... and label them with the respective list entries
    ax.set_xticklabels(actions)
    ax.set_yticklabels(range(total_number_of_states))

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    # for i in range(total_number_of_states):
        # for j in range(total_number_of_actions):
            # ax.text(j, i, int(q.Q_table[i, j]*100), ha="center", va="center", color="w")

    ax.set_title("Q_table")
    fig.tight_layout()
    plt.show()


def count(test_list, value):
    number_of_occurrences = 0
    for element in test_list:
        if element == value:
            number_of_occurrences = number_of_occurrences + 1
    return number_of_occurrences


class QLearning:
    def __init__(self, index):
        # Parameters for the algorithm
        self.learning_rate = 0.7
        self.discount_factor = 0.34
        self.explore_rate = 0.2
        self.sum_of_rewards = 0.0
        self.training = 1  # determine if the q table is updated and if there is going to be any explorations.
        self.Q_table = np.zeros((total_number_of_states, total_number_of_actions))

        # Parameters for the interpretations of the game
        self.player_index = index
        self.current_state = [0, 0, 0, 0]
        self.last_action = no_action
        self.last_player_pieces = [0, 0, 0, 0]
        self.number_of_wins = 0
        self.number_of_games = 0
        self.last_state = 0
        self.number_of_steps = 0

    def reset_game(self):
        self.current_state = [0, 0, 0, 0]
        self.last_action = no_action
        self.last_state = 0
        self.last_player_pieces = [0, 0, 0, 0]
        self.sum_of_rewards = 0.0
        self.number_of_games = self.number_of_games + 1
        self.number_of_steps = 0

    def determined_state(self, player_pieces):
        return player_pieces

    def determined_possible_actions(self, player_pieces, enemy_pieces, dice):
        possible_actions = [normal_action, normal_action, normal_action, normal_action]

        for piece_index in range(number_of_pieces):
            old_piece_pos = player_pieces[piece_index]
            new_piece_pos = old_piece_pos + dice
            enemy_at_pos, enemy_pieces_at_pos = player.get_enemy_at_pos(new_piece_pos, enemy_pieces)
            if old_piece_pos == player.GOAL_INDEX or (old_piece_pos == player.HOME_INDEX and dice < 6):  # piece at goal
                possible_actions[piece_index] = no_action

            elif old_piece_pos == player.HOME_INDEX and dice == 6:  # move out of home
                possible_actions[piece_index] = move_out_action

            elif new_piece_pos in player.STAR_INDEXS:  # use a star to jump
                possible_actions[piece_index] = use_star_action

            elif new_piece_pos == player.GOAL_INDEX or new_piece_pos == player.STAR_AT_GOAL_AREAL_INDX:  # enter goal
                possible_actions[piece_index] = enter_goal_action

            elif new_piece_pos in player.GLOB_INDEXS:  # globe not owned
                if enemy_at_pos != player.NO_ENEMY:
                    possible_actions[piece_index] = suicide_action
                else:
                    possible_actions[piece_index] = move_to_safety_action
            elif new_piece_pos in player.LIST_ENEMY_GLOB_INDEX:
                # Get the enemy their own the glob
                globs_enemy = player.LIST_TAILE_ENEMY_GLOBS.index(player.BORD_TILES[new_piece_pos])
                # Check if there is an enemy at the glob
                if enemy_at_pos != player.NO_ENEMY:
                    # If there is another enemy then send them home and move there
                    if enemy_at_pos != globs_enemy:
                        possible_actions[piece_index] = kill_enemy_action
                    # If it is the same enemy that is there then move there
                    else:
                        possible_actions[piece_index] = suicide_action
                # If there are not any enemy's at the glob then move there
                else:
                    possible_actions[piece_index] = move_to_safety_action

            elif enemy_at_pos != player.NO_ENEMY: # kill or suicide
                if len(enemy_pieces_at_pos) == 1:
                    possible_actions[piece_index] = kill_enemy_action
                else:
                    possible_actions[piece_index] = suicide_action

            elif old_piece_pos < 53 and new_piece_pos > 52: # goal zone
                possible_actions[piece_index] = enter_goal_zone_action

            elif old_piece_pos in player.GLOB_INDEXS or count(player_pieces, old_piece_pos) > 1:
                possible_actions[piece_index] = move_away_from_safe_action

            elif new_piece_pos > 52 and not (new_piece_pos == player.GOAL_INDEX):
                possible_actions[piece_index] = in_goal_zone_action
            else:
                possible_actions[piece_index] = normal_action
        return possible_actions

    def get_reward(self, player_pieces, there_is_a_winner):
        reward = 0.0

        if self.last_action == move_out_action:
            reward += 0.3

        elif self.last_action == kill_enemy_action:
            reward += 0.2

        elif self.last_action == suicide_action:
            reward += -0.8

        elif self.last_action == normal_action:
            reward += 0.05

        elif self.last_action == in_goal_zone_action:
            reward += 0.05

        elif self.last_action == enter_goal_zone_action:
            reward += 0.2

        elif self.last_action == use_star_action:
            reward += 0.15

        elif self.last_action == enter_goal_action:
            reward += 0.25

        elif self.last_action == move_away_from_safe_action:
            reward += -0.1

        elif self.last_action == move_to_safety_action:
            reward += 0.1

        elif self.last_action == no_action:
            reward += -0.1

        for i in range(number_of_pieces):
            if self.last_player_pieces[i] > 0 and player_pieces[i] == 0: # Means that the pieces have been moved home
                reward += -0.25
                break

        if not(self.last_action == move_out_action or self.last_action == in_goal_zone_action or
                self.last_action == enter_goal_action or self.last_action == enter_goal_zone_action or
                self.last_action == suicide_action or self.last_action == no_action):
            lowest_index = -1
            lowest_value = 100
            for i in range(number_of_pieces):                           # Used to check if the piece closest
                if not(self.last_player_pieces[i] == 0):                # to home has been moved
                    if self.last_player_pieces[i] < lowest_value:
                        lowest_value = self.last_player_pieces[i]
                        lowest_index = i
            if not(lowest_index == -1):
                if self.last_player_pieces[lowest_index] < player_pieces[i]:
                    reward += 0.1

        player_won = True
        if there_is_a_winner:                          # check if the agent won the game.
            for i in range(number_of_pieces):
                if not player_pieces[i] == player.GOAL_INDEX:
                    player_won = False
                    break
            if player_won:
                reward += 1.0
            else:
                reward += -1.0

        return reward

    def pick_action(self,piece_states,piece_actions):
        best_action_player = -1
        if not (piece_actions.count(no_action) == len(piece_actions)):
            if self.explore_rate == 0 or  np.random.random() > self.explore_rate:
                max_q = -1000.
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

    def pick_action_max(self,piece_states,piece_actions):
        best_action_player = -1
        if not (piece_actions.count(no_action) == len(piece_actions)):
            max_q = -1000.
            for i in range(4):
                if not(piece_actions[i] == no_action):
                    if max_q < self.Q_table[piece_states[i]][piece_actions[i]]:
                        max_q = self.Q_table[piece_states[i]][piece_actions[i]]
                        best_action_player = i

        return best_action_player

    def update_q_table(self, player_pieces, enemy_pieces, dice, game, there_is_a_winner):
        current_actions = self.determined_possible_actions( player_pieces,enemy_pieces,dice)
        current_states = self.determined_state(player_pieces)
        piece_index = self.pick_action(current_states, current_actions)
        if self.training == 1 and not(piece_index == -1):

            reward = self.get_reward(player_pieces, there_is_a_winner)
            self.sum_of_rewards += reward

            max_piece_index = self.pick_action_max(current_states, current_actions)
            max_q_value_next_state = self.Q_table[current_states[max_piece_index]][current_actions[max_piece_index]]

            last_q_value = self.Q_table[self.last_state][self.last_action]

            self.Q_table[self.last_state][self.last_action] += \
                self.learning_rate*(reward+self.discount_factor*max_q_value_next_state-last_q_value)

            self.last_player_pieces = player_pieces
            self.last_state = current_states[piece_index]
            self.last_action = current_actions[piece_index]
            self.number_of_steps = self.number_of_steps + 1
        return piece_index

    def save_Q_table(self,file_name):
        file_ext = file_name.split(".")[-1]
        assert file_ext == "npy", "The file extension has to be npy (numpy file)"
        np.save(file_name, self.Q_table)

    def load_Q_table(self,file_name):
        file_ext = file_name.split(".")[-1]
        assert file_ext == "npy", "The file extension has to be npy (numpy file)"
        self.Q_table = np.load(file_name)

