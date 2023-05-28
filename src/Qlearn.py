import sys
sys.path.insert(0, "/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project")
import numpy as np
from ludopy import player
import os
from enum import Enum

def count(list, value):
    no_Ocurrences = 0;
    for element in list:
        if element == value:
            no_Ocurrences = no_Ocurrences + 1
    return no_Ocurrences


class Action(Enum):
    STARTING_ACTION = 0
    DEFAULT_ACTION = 1
    INSIDE_GOAL_AREA_ACTION = 2
    ENTER_GOAL_AREA_ACTION = 3
    ENTER_WINNING_AREA_ACTION = 4
    STAR_ACTION = 5
    MOVE_INSIDE_SAFETY_ACTION = 6
    MOVE_OUTSIDE_SAFETY_ACTION = 7
    KILL_PLAYER_ACTION = 8
    DIE_ACTION = 9
    NO_ACTION = 10

class State(Enum):
    START_AREA = 0
    GOAL_AREA = 1
    WINNING_AREA = 2
    DANGER_AREA = 3
    SAFE_AREA = 4
    DEFAULT_AREA = 5
    

class QLearn:
    def __init__(self, index):
        # Number of game bricks 
        self.no_gameBricks = 4
        self.number_States = 6
        self.number_Actions = 11

        self.learning_rate = 0.5
        self.discount_factor = 0.5
        self.boltzmann_temperature = 0.5
        self.sum_of_rewards = 0.0
        self.training = 1 
        self.Q_table = np.zeros((self.number_States, self.number_Actions), dtype=float)

        self.player_index = index
        self.number_of_wins = 0
        self.number_of_games = 0
        self.actions_per_game = []
        self.actions_this_game = 0 
        
        # Initialize reset
        self.reset()


    def reset(self):
        self.current_state = [State.START_AREA.value] * 4
        self.last_action = None
        self.last_state = State.START_AREA.value
        self.last_player_pieces = [0] * 4
        self.sum_of_rewards = 0.0
        self.number_of_games += 1
        self.actions_per_game.append(self.actions_this_game)
        self.actions_this_game = 0 

    # State logic
    def pieceState(self, piece_pos, player_pieces, enemy_pieces, game):
        if piece_pos == player.HOME_INDEX:
            return State.START_AREA.value
        elif piece_pos in player.HOME_AREAL_INDEXS:
            return State.GOAL_AREA.value
        elif piece_pos == player.GOAL_INDEX:
            return State.WINNING_AREA.value
        elif piece_pos in player.GLOB_INDEXS or count(player_pieces, piece_pos) > 1:
            return State.SAFE_AREA.value
        else:
            return self.otherState(piece_pos, player_pieces, enemy_pieces, game)

    def otherState(self, piece_pos, player_pieces, enemy_pieces, game):
        for enemy_index, enemy_glob in enumerate(player.LIST_ENEMY_GLOB_INDEX1):
            if piece_pos == enemy_glob and player.HOME_INDEX in enemy_pieces[enemy_index]:
                if enemy_index not in game.ghost_players:
                    return State.DANGER_AREA.value
                else:
                    return State.SAFE_AREA.value

        return self.enemyInRange(piece_pos, player_pieces, enemy_pieces)

    def enemyInRange(self, piece_pos, player_pieces, enemy_pieces):
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
                return State.DANGER_AREA.value

        return State.DEFAULT_AREA.value
    
    def statesLogic(self, player_pieces, enemy_pieces, game):
        state_of_pieces = [self.pieceState(piece_pos, player_pieces, enemy_pieces, game)
                           for piece_pos in player_pieces]
        return state_of_pieces

    #Action logic
    def actionsLogic(self, player_pieces, enemy_pieces, dice):
        possible_actions = []
        
        for piece_index, old_piece_pos in enumerate(player_pieces):
            new_piece_pos = old_piece_pos + dice
            enemy_at_pos, enemy_pieces_at_pos = player.get_enemy_at_pos(new_piece_pos, enemy_pieces)
            
            possible_actions.append(self.pieceAction(old_piece_pos, new_piece_pos, dice, enemy_at_pos, enemy_pieces_at_pos, player_pieces))
        
        return possible_actions

    def pieceAction(self, old_piece_pos, new_piece_pos, dice, enemy_at_pos, enemy_pieces_at_pos, player_pieces):
        # Piece at goal   
        if old_piece_pos == player.GOAL_INDEX or (old_piece_pos == player.HOME_INDEX and dice < 6):
            return Action.NO_ACTION.value
        # Move out of home
        elif old_piece_pos == player.HOME_INDEX and dice == 6:
            return Action.STARTING_ACTION.value
        # Use a star to jump
        elif new_piece_pos in player.STAR_INDEXS:
            return Action.STAR_ACTION.value
        # Enter goal
        elif new_piece_pos == player.GOAL_INDEX or new_piece_pos == player.STAR_AT_GOAL_AREAL_INDX:
            return Action.ENTER_WINNING_AREA_ACTION.value
        # Globe not owned
        elif new_piece_pos in player.GLOB_INDEXS and enemy_at_pos == player.NO_ENEMY:
            return Action.MOVE_INSIDE_SAFETY_ACTION.value
        elif new_piece_pos in player.GLOB_INDEXS and enemy_at_pos != player.NO_ENEMY:
            return Action.DIE_ACTION.value
        # Piece on a globe on top of another piece
        elif (count(player_pieces, new_piece_pos) > 0 or old_piece_pos in player.GLOB_INDEXS or 
            count(player_pieces, old_piece_pos) > 1):
            return Action.MOVE_INSIDE_SAFETY_ACTION.value
        # Globe owned by an enemy 
        elif self.enemyGlobe(new_piece_pos, enemy_at_pos, player_pieces):
            return Action.MOVE_INSIDE_SAFETY_ACTION.value
        elif self.enemyGlobe(new_piece_pos, enemy_at_pos, player_pieces) == False:
            return Action.DIE_ACTION.value
        # Enemy at position
        elif self.enemyPosition(enemy_at_pos, enemy_pieces_at_pos):
            return Action.KILL_PLAYER_ACTION.value
        elif self.enemyPosition(enemy_at_pos, enemy_pieces_at_pos) == False:
            return Action.DIE_ACTION.value
        # Enter goal area
        elif old_piece_pos < 53 and new_piece_pos > 52:
            return Action.ENTER_GOAL_AREA_ACTION.value
        # Move outside safety
        else:
            return Action.MOVE_OUTSIDE_SAFETY_ACTION.value

    def enemyGlobe(self, new_piece_pos, enemy_at_pos, player_pieces):
        if new_piece_pos in player.LIST_ENEMY_GLOB_INDEX1 and enemy_at_pos == player.NO_ENEMY:
            return True
        elif new_piece_pos in player.LIST_ENEMY_GLOB_INDEX1 and enemy_at_pos != player.NO_ENEMY:
            globs_enemy = player.LIST_TAILE_ENEMY_GLOBS.index(player.BORD_TILES[new_piece_pos])
            if enemy_at_pos != globs_enemy:
                return True
        return False

    def enemyPosition(self, enemy_at_pos, enemy_pieces_at_pos):
        if enemy_at_pos != player.NO_ENEMY and len(enemy_pieces_at_pos) == 1:
            return True
        elif enemy_at_pos != player.NO_ENEMY and len(enemy_pieces_at_pos) > 1:
            return False
        return None

    def reward(self, player_pieces, current_states, there_is_a_winner):
        reward = 0.0

        if self.last_action == Action.STARTING_ACTION.value:
            reward += 0.25
        if self.last_action == Action.KILL_PLAYER_ACTION.value:
            reward = -0.7
        if self.last_action == Action.DIE_ACTION.value:
            reward += -0.2
        if self.last_action == Action.DEFAULT_ACTION.value:
            reward += 0.001
        if self.last_action == Action.INSIDE_GOAL_AREA_ACTION.value:
            reward += 0.4
        if self.last_action == Action.ENTER_GOAL_AREA_ACTION.value:
            reward += 0.4
        if self.last_action == Action.STAR_ACTION.value:
            reward += 0.6
        if self.last_action == Action.ENTER_WINNING_AREA_ACTION.value:
            reward += 1.0
        if self.last_action == Action.MOVE_OUTSIDE_SAFETY_ACTION.value:
            reward += -0.4
        if self.last_action == Action.MOVE_INSIDE_SAFETY_ACTION.value:
            reward += 0.3
        if self.last_action == Action.NO_ACTION.value:
            reward += -0.05

        for i in range(self.no_gameBricks):
            if self.last_player_pieces[i] > 0 and player_pieces[i] == 0: 
                # A piece has been moved home
                reward += -0.5
                break

        # # Reward logic based on current states
        for state in current_states:
            if state == State.DANGER_AREA.value:
                reward -= 0.1  
            elif state == State.WINNING_AREA.value:
                reward += 0.1 

        # Reward logic based on there is a winner
        if there_is_a_winner:
            if there_is_a_winner == self.player_index:
                reward += 1.0 
            else:
                reward -= 1.0


        return reward

    # Action selection logic using Boltzmann Exploration
    def pickAction(self, pieceStates, piece_actions):
        if not (piece_actions.count(Action.NO_ACTION.value) == len(piece_actions)):
            valid_actions = [i for i in range(4) if piece_actions[i] != Action.NO_ACTION.value]
            q_values = np.array([self.Q_table[pieceStates[i]][piece_actions[i]] for i in valid_actions])
            action_probs = np.exp(q_values / self.boltzmann_temperature) / np.sum(np.exp(q_values / self.boltzmann_temperature))
            best_action_player = np.random.choice(valid_actions, p=action_probs)
        else:
            best_action_player = -1
        return best_action_player

    # Update Q-table logic    
    def updateQLogic(self, player_pieces, enemy_pieces, dice, game, there_is_a_winner):
        current_actions = self.actionsLogic(player_pieces,enemy_pieces,dice)
        current_states = self.statesLogic(player_pieces, enemy_pieces, game)
        piece_index = self.pickAction(current_states, current_actions)
        if self.training == 1 and piece_index is not None:
            self.actions_this_game += 1  # Increment the action count each time an action is performed
            reward = self.reward(player_pieces, current_states, there_is_a_winner)
            self.sum_of_rewards += reward
            current_q_value = self.Q_table[current_states[piece_index]][current_actions[piece_index]]
            last_q_value = self.Q_table[self.last_state][self.last_action]
            self.Q_table[self.last_state][self.last_action] += \
                self.learning_rate*(reward+self.discount_factor*current_q_value-last_q_value)
            self.last_player_pieces = player_pieces
            self.last_state = current_states[piece_index]
            self.last_action = current_actions[piece_index]
            
        return piece_index

    def save_QTable(self,file_name):
        folder_path = os.path.join(os.getcwd(), "/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project/src/data")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        data_file_path = os.path.join(folder_path, file_name)
        np.save(data_file_path, self.Q_table)

    def load_QTable(self,file_name):
            file_ext = file_name.split(".")[-1]
            assert file_ext == "npy", "The file extension has to be npy (numpy file)"
            folder_path = "/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project/src/data"
            data_file_path = os.path.join(folder_path, file_name)
            self.Q_table = np.load(data_file_path)
