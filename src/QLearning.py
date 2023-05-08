import sys
sys.path.insert(0, "/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project")
import numpy as np
from ludopy import player
import os

def count(list, value):
    no_Ocurrences = 0;
    for element in list:
        if element == value:
            no_Ocurrences = no_Ocurrences + 1
    return no_Ocurrences


class QLearn:
    def __init__(self, index):
        # Number of game bricks 
        self.no_gameBricks = 4
        self.number_States = 6
        self.number_Actions = 11

        # Initialize states, actions and parameters
        self.intializeStates()
        self.initializeActions()
        self.initializeParameters(index)
        
        # Initialize reset
        self.reset()

    def intializeStates(self):
        self.startArea = 0
        self.goalArea = 1
        self.winningArea = 2
        self.dangerArea = 3
        self.safeArea = 4
        self.defaultArea = 5
        
    def initializeActions(self):
        self.starting_action = 0
        self.default_action = 1
        self.inside_goalArea_action = 2
        self.enter_goalArea_action = 3
        self.enter_winningArea_action = 4
        self.star_action = 5
        self.move_inside_safety_action = 6
        self.move_outside_safety_action = 7
        self.kill_player_action = 8
        self.die_action = 9
        self.no_action = 10

    def initializeParameters(self, index):
        self.learning_rate = 0.7  # alpha
        self.discount_factor = 0.4  # gama pre 0.01
        self.explore_rate = 0.4  # epsilon
        self.sum_of_rewards = 0.0
        self.training = 1  # determineds if the q table is updated and if there is going to be any explorations.
        self.Q_table = np.zeros((self.number_States, self.number_Actions), dtype=float)


        self.player_index = index
        self.number_of_wins = 0
        self.number_of_games = 0


    def reset(self):
        self.current_state = [self.startArea] * 4
        self.last_action = None
        self.last_state = self.startArea
        self.last_player_pieces = [0] * 4
        self.sum_of_rewards = 0.0
        self.number_of_games += 1

    # State determination logic
    def determined_state(self, player_pieces, enemy_pieces, game):
        state_of_pieces = [self.startArea] * self.no_gameBricks
        
        for piece_index, piece_pos in enumerate(player_pieces):
            if piece_pos == player.HOME_INDEX:
                state_of_pieces[piece_index] = self.startArea
            elif piece_pos in player.HOME_AREAL_INDEXS:
                state_of_pieces[piece_index] = self.goalArea
            elif piece_pos == player.GOAL_INDEX:
                state_of_pieces[piece_index] = self.winningArea
            elif piece_pos in player.GLOB_INDEXS or count(player_pieces, piece_pos) > 1:
                state_of_pieces[piece_index] = self.safeArea
            else:
                state_determined = False
                for enemy_index, enemy_glob in enumerate(player.LIST_ENEMY_GLOB_INDEX1):
                    if piece_pos == enemy_glob and player.HOME_INDEX in enemy_pieces[enemy_index]:
                        if enemy_index not in game.ghost_players:
                            state_of_pieces[piece_index] = self.dangerArea
                        else:
                            state_of_pieces[piece_index] = self.safeArea
                        state_determined = True
                        break
                if not state_determined:
                    if piece_pos in player.STAR_INDEXS:
                        if piece_pos in player.STAR_INDEXS[1::2]:
                            range_to_look_for_enemies = list(range(1, 7)) + list(range(8, 14))
                        else:
                            range_to_look_for_enemies = list(range(1, 13))
                    else:
                        range_to_look_for_enemies = list(range(1, 7))
                    
                    for index in range_to_look_for_enemies:
                        enemy_pos = (piece_pos - index) % 52
                        enemy_at_pos, _ = player.get_enemy_at_pos(enemy_pos, enemy_pieces)
                        if enemy_at_pos != player.NO_ENEMY:
                            state_of_pieces[piece_index] = self.dangerArea
                            break
                    else:
                        state_of_pieces[piece_index] = self.defaultArea
        
        return state_of_pieces


    # Possible action determination logic
    def determined_actions(self, player_pieces, enemy_pieces, dice):
        possible_actions = []
        for piece_index, old_piece_pos in enumerate(player_pieces):
            new_piece_pos = old_piece_pos + dice

            # Piece at goal
            if old_piece_pos == player.GOAL_INDEX or (old_piece_pos == player.HOME_INDEX and dice < 6):
                possible_actions.append(self.no_action)
                continue

            # Move out of home
            if old_piece_pos == player.HOME_INDEX and dice == 6:
                possible_actions.append(self.starting_action)
                continue

            # Use a star to jump
            if new_piece_pos in player.STAR_INDEXS:
                possible_actions.append(self.star_action)
                continue

            # Enter goal
            if new_piece_pos == player.GOAL_INDEX or new_piece_pos == player.STAR_AT_GOAL_AREAL_INDX:
                possible_actions.append(self.enter_winningArea_action)
                continue

            # Globe not owned
            if new_piece_pos in player.GLOB_INDEXS:
                enemy_at_pos, enemy_pieces_at_pos = player.get_enemy_at_pos(new_piece_pos, enemy_pieces)
                if enemy_at_pos != player.NO_ENEMY:
                    possible_actions.append(self.die_action)
                else:
                    possible_actions.append(self.move_inside_safety_action)
                continue

            # Piece on a globe or stacked on top of another piece
            if count(player_pieces, new_piece_pos) > 0 or old_piece_pos in player.GLOB_INDEXS or count(player_pieces, old_piece_pos) > 1:
                possible_actions.append(self.move_inside_safety_action)
                continue

            # Globe owned by an enemy
            if new_piece_pos in player.LIST_ENEMY_GLOB_INDEX1:
                globs_enemy = player.LIST_TAILE_ENEMY_GLOBS.index(player.BORD_TILES[new_piece_pos])
                enemy_at_pos, enemy_pieces_at_pos = player.get_enemy_at_pos(new_piece_pos, enemy_pieces)
                if enemy_at_pos != player.NO_ENEMY:
                    if enemy_at_pos != globs_enemy:
                        possible_actions.append(self.kill_player_action)
                    else:
                        possible_actions.append(self.die_action)
                else:
                    possible_actions.append(self.move_inside_safety_action)
                continue

            # Enemy at position
            enemy_at_pos, enemy_pieces_at_pos = player.get_enemy_at_pos(new_piece_pos, enemy_pieces)
            if enemy_at_pos != player.NO_ENEMY:
                if len(enemy_pieces_at_pos) == 1:
                    possible_actions.append(self.kill_player_action)
                else:
                    possible_actions.append(self.die_action)
                continue

            # Enter goal area
            if old_piece_pos < 53 and new_piece_pos > 52:
                possible_actions.append(self.enter_goalArea_action)
                continue

            # Move outside safety
            possible_actions.append(self.move_outside_safety_action)

        return possible_actions

    def getReward(self, player_pieces, current_states, there_is_a_winner):
        reward = 0.0

        if self.last_action == self.starting_action:
            reward += 0.1
        if self.last_action == self.kill_player_action:
            reward += 0.2
        if self.last_action == self.die_action:
            reward += -0.1
        if self.last_action == self.default_action:
            reward += 0.05
        if self.last_action == self.inside_goalArea_action:
            reward += 0.2
        if self.last_action == self.enter_goalArea_action:
            reward += 0.5
        if self.last_action == self.star_action:
            reward += 0.15
        if self.last_action == self.enter_winningArea_action:
            reward += 1.0
        if self.last_action == self.move_outside_safety_action:
            reward += -0.1
        if self.last_action == self.move_inside_safety_action:
            reward += 0.1
        if self.last_action == self.no_action:
            reward += -0.05

        for i in range(self.no_gameBricks):
            if self.last_player_pieces[i] > 0 and player_pieces[i] == 0: 
                # A piece has been moved home
                reward += -0.5
                break

        return reward



    # Action selection logic
    def pick_action(self, piece_states, piece_actions):
        temperature = 0.5  # Set the temperature parameter for the Softmax function
        if not (piece_actions.count(self.no_action) == len(piece_actions)):
            valid_actions = [i for i in range(4) if piece_actions[i] != self.no_action]
            q_values = np.array([self.Q_table[piece_states[i]][piece_actions[i]] for i in valid_actions])
            action_probs = np.exp(q_values / temperature) / np.sum(np.exp(q_values / temperature))
            best_action_player = valid_actions[np.random.choice(np.arange(len(action_probs)), p=action_probs)]
        else:
            best_action_player = -1
        return best_action_player



    # Update Q-table logic    
    def updateQTable(self, player_pieces, enemy_pieces, dice, game, there_is_a_winner):
        current_actions = self.determined_actions(player_pieces,enemy_pieces,dice)
        current_states = self.determined_state(player_pieces, enemy_pieces, game)
        piece_index = self.pick_action(current_states, current_actions)
        if self.training == 1 and piece_index is not None:
            reward = self.getReward(player_pieces, current_states, there_is_a_winner)
            self.sum_of_rewards += reward
            current_q_value = self.Q_table[current_states[piece_index]][current_actions[piece_index]]
            last_q_value = self.Q_table[self.last_state][self.last_action]
            self.Q_table[self.last_state][self.last_action] += \
                self.learning_rate*(reward+self.discount_factor*current_q_value-last_q_value)
            self.last_player_pieces = player_pieces
            self.last_state = current_states[piece_index]
            self.last_action = current_actions[piece_index]
            
            # Learning rate decay (Future work)
            #     # Check if Q-value update is below threshold
            # if abs(delta) < threshold:
            #     self.training = 0  # Stop updating Q-table
            
        return piece_index

    def save_QTable(self,file_name):
        folder_path = os.path.join(os.getcwd(), "/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project/src/data")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        data_file_path = os.path.join(folder_path, file_name)
        np.save(data_file_path, self.Q_table)

    def load_Q_table(self,file_name):
            file_ext = file_name.split(".")[-1]
            assert file_ext == "npy", "The file extension has to be npy (numpy file)"
            folder_path = "/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project/src/data"
            data_file_path = os.path.join(folder_path, file_name)
            self.Q_table = np.load(data_file_path)
