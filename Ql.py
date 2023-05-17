import copy
import numpy as np
import ludopy


class Qplayer:
    def __init__(self, tableName="QLearn/Qs.txt", newtbl=False, isTrain=False):
        self.tableName = tableName
        self.LUT = self.Qinit(newtbl)
        self.prev_state = np.zeros([4], dtype=int)
        self.prev_action = 9*np.ones([4], dtype=int)
        # tiles is the type of square, list goes [globe, star, goalzone, goal]
        self.tiles = [[1, 9, 22, 35, 48, 53], [5, 12, 18, 25, 31, 38, 44, 51], [53, 54, 55, 56, 57, 58], [59]]
        self.istrain = isTrain

        # discount
        self.delta = 0.3

    def nextmove(self, player_i, player_pieces, enemy_pieces, dice, move_pieces):
        self.piece, self.action, self.state = self.map2gamespace(player_i, player_pieces,
                                                                 enemy_pieces, dice, move_pieces)

    def Qinit(self, new):
        num_s = 16
        num_a = 10

        if new == 0:
            LUT = np.loadtxt(self.tableName, delimiter=' ')
        else:
            LUT = np.zeros([num_s, num_a])

        return LUT

    def findstate(self, i, possiblevec, player_pieces, enemy_pieces):
        home = 0
        goal_zone = [53, 54, 55, 56, 57, 58, 59]
        safe = [1, 9, 22, 35, 48]
        danger = [14, 27, 40]

        adj_enemy = self.adjustenemy(enemy_pieces)

        if player_pieces[i] == home:
            state = 0 + 4 * i
        elif self.ismember(goal_zone, player_pieces[i]) == 1:
            state = 1 + 4 * i
            possiblevec[5, 5] = 0
        elif self.ismember(safe, player_pieces[i]) == 1:
            state = 2 + 4 * i
        else:
            if self.iswithin(player_pieces[i],adj_enemy) or self.ismember(danger, player_pieces[i]):
                state = 3 + 4 * i
            else:
                state = 2 + 4 * i

        return state

    def iswithin(self, player, enemy_pieces):

        if self.ismember(enemy_pieces, player-1):
            return 1
        elif self.ismember(enemy_pieces, player-2):
            return 1
        elif self.ismember(enemy_pieces, player-3):
            return 1
        elif self.ismember(enemy_pieces, player-4):
            return 1
        elif self.ismember(enemy_pieces, player-5):
            return 1
        elif self.ismember(enemy_pieces, player-6):
            return 1
        else:
            return 0

    def map2gamespace(self, player_id, player_pieces, enemy_pieces, dice, active_pieces):

        num_pieces = len(active_pieces)
        action_index = -1*np.ones([2, 4], dtype=float)
        state = np.zeros([4], dtype=int)

        if num_pieces == 0:
            piece = -1
            action = 9
            state = 0 + 4 * player_id
            return piece, action, state

        for i in range(0, num_pieces):
            i = active_pieces[i]
            possiblevec = self.possiblemoves(player_pieces[i], player_pieces, enemy_pieces, dice)
            state[i] = self.findstate(i, possiblevec, player_pieces,enemy_pieces)

            moves = self.LUT[state[i]] * possiblevec
            moves = np.diag(moves)
            possiblevec = np.diag(possiblevec)
            possiblevec = np.array(np.where(possiblevec != 0)).ravel()

            if sum(moves) == 0:
                action_index[0, i] = np.random.choice(possiblevec)
            else:
                action_index[0, i] = np.argmax(moves)
            action_index[1, i] = self.LUT[state[i], int(action_index[0, i])]
            action_index[0, i], action_index[1, i] = self.isexploration(possiblevec, action_index[0, i], action_index[1, i])

        if (dice == 6) & (self.ismember(player_pieces, 0) >= 1):
            piece = np.where(player_pieces == 0)[0]
            action = 0
            piece = piece[np.random.randint(0, len(piece))]
        else:
            piece = action_index[1].argmax()
            action = action_index[0][piece]

        return int(piece), int(action), int(state[piece])

    def adjustenemy(self, enemy):
        adj_enemy = np.zeros(enemy.shape)

        for i in range(enemy.shape[0]):
            for j in range(enemy.shape[1]):
                if (enemy[i, j] != 0) & (enemy[i, j] < 54):
                    adj_enemy[i, j] = enemy[i, j] + (i + 1) * 13
                    if (adj_enemy[i, j] / 53) >= 1:
                        adj_enemy[i, j] += -52
        return adj_enemy

    def possiblemoves(self, current_piece, player_pieces, enemy, dice):
        vecPossibleMoves = np.zeros([10, 10])
        # tiles is the type of square, list goes [globe, star, goalzone, goal]
        tiles = [[1, 9, 22, 35, 48, 53], [5, 12, 18, 25, 31, 38, 44, 51], [53, 54, 55, 56, 57, 58],
                 [59]]

        adj_enemy = self.adjustenemy(enemy)

        if current_piece != 0:
            if tiles[3] == (current_piece + dice):
                vecPossibleMoves[2][2] = 1
            if self.ismember(tiles[1], current_piece + dice):
                vecPossibleMoves[3][3] = 1
            if self.ismember(tiles[0], current_piece + dice):
                vecPossibleMoves[4][4] = 1
                if (current_piece + dice) == tiles[1][-1]:
                    vecPossibleMoves[2][2] = 1
            if self.ismember(player_pieces, current_piece + dice):
                vecPossibleMoves[5][5] = 1
            if self.ismember(adj_enemy.ravel(), current_piece + dice) == 1:
                vecPossibleMoves[6][6] = 1
            if (current_piece + dice) > 59:
                vecPossibleMoves[7][7] = 1
            if self.ismember(tiles[2], current_piece + dice):
                vecPossibleMoves[8][8] = 1
            if sum(np.diag(vecPossibleMoves)) == 0:
                vecPossibleMoves[1][1] = 1
        else:
            if dice == 6:
                vecPossibleMoves[0][0] = 1
            else:
                vecPossibleMoves[9][9] = 1

        return vecPossibleMoves

    def isexploration(self, possiblemoves, og_action, og_qvalue):
        if self.istrain:
            explore_rate = 0.1
            explore_chance = np.random.random()
            if explore_chance > (1 - explore_rate):
                a = np.random.choice(possiblemoves)
                q = 10
                return a, q
            else:
                return og_action, og_qvalue
        else:
            return og_action, og_qvalue

    def reward(self, max_Q_new, piece):
        def zero(self, piece):
            # Open
            r = 0.25
            return r

        def one(self, piece):
            # Normal move
            r = 0.0001
            return r

        def two(self, piece):
            # Goal move
            r = 0.5
            return r

        def three(self, piece):
            # Star Move
            r = 0.6
            return r

        def four(self, piece):
            # Globe Move
            r = 0.4
            return r

        def five(self, piece):
            # Protect move
            r = 0.2
            return r

        def six(self, piece):
            # Kill move
            r = 0.9
            return r

        def seven(self, piece):
            # Backwards goal zone move
            r = 0
            return r

        def eight(self, piece):
            # Goal Zone Move
            r = 0.4
            return r


        switcher = {
            0: zero,
            1: one,
            2: two,
            3: three,
            4: four,
            5: five,
            6: six,
            7: seven,
            8: eight,
        }

        LR = 0.1  # learning rate
        func = switcher.get(self.action, lambda: "Invalid Action")
        r = func(self, piece)
        if (self.state % 3) == 0:
            r += 0.1
        delta_LUT = LR * (r + self.delta * max_Q_new - self.LUT[self.state, self.action])
        self.LUT[self.state, self.action] += delta_LUT

    def train(self, new_player_pieces, new_enemy_pieces):
        if self.istrain:
            new_actions = np.zeros([2, 6])
            if not (self.piece == -1):
                for i in range(1, 7):
                    possiblemoves = self.possiblemoves(new_player_pieces[self.piece], new_player_pieces,
                                                       new_enemy_pieces, i)
                    pp = copy.deepcopy(new_player_pieces)
                    pp[self.piece] += i
                    new_actions[1, i-1] = self.findstate(self.piece, possiblemoves, pp, new_enemy_pieces)
                    new_actions[0, i-1] = self.LUT[int(new_actions[1, i-1]), np.argmax(np.diag(possiblemoves))]
                max_Q_new = np.max(new_actions[0])
                self.reward(max_Q_new, new_player_pieces[self.piece])
                self.prev_state[self.piece] = self.state
                self.prev_action[self.piece] = self.action
        else:
            pass

    def write2text(self):
        np.savetxt(self.tableName, self.LUT, delimiter=' ', fmt='%1.4f')

    def ismember(self, A, B):
        w = [np.sum(a == B) for a in A]
        w = np.sum(w)
        return w

    def averageqs(self):

        for ii in range(0, 9):
            for jj in range(0, 4):
                temp = 0
                for kk in range(0, 4):
                    temp += self.LUT[jj+kk*4, ii]
                temp = temp/4
                self.LUT[jj, ii] = temp
                self.LUT[jj + 4, ii] = temp
                self.LUT[jj + 8, ii] = temp
                self.LUT[jj + 12, ii] = temp


def QLearn(players:list) -> tuple[int, int, int]:
    def tuple_to_dict(observation: tuple) -> dict:
        _obs = {
            "dice": observation[0],
            "move_pieces": observation[1],
            "player_pieces": observation[2],
            "enemy_pieces": observation[3],
            "player_is_a_winner": observation[4],
            "there_is_a_winner": observation[5]
        }
        return _obs

    _obs: tuple
    _current_player: int
    
    _actions: int = 0
    _overshoots: int = 0

    _game = ludopy.Game(players)

    while True:
        _obs, _current_player = _game.get_observation()
        _obs = tuple_to_dict(_obs)

        if type(players[_current_player]) == type(ludopy.player.Player()):
            _move_pieces = _obs.get("move_pieces")
            if len(_move_pieces):
                _piece_to_move = _move_pieces[np.random.randint(0, len(_move_pieces))]
                if _obs.get("player_pieces")[_piece_to_move]+  _obs.get("dice") > 59:
                    _overshoots += 1
            else:
                _piece_to_move = -1
            _game.answer_observation(_piece_to_move)
        else:
            players[_current_player].nextmove(_current_player, _obs.get("player_pieces"), _obs.get("enemy_pieces"), _obs.get("dice"), _obs.get("move_pieces"))
            _piece_to_move = players[_current_player].piece
            _, _, new_P0, new_enemy, _, _ = _game.answer_observation(_piece_to_move)
            players[_current_player].train(new_P0, new_enemy)
            if players[_current_player].action == 7:
                _actions += 1


        if _game.get_winner_of_game() != -1:
            break
        if _game.all_players_finish():
            break

    _overshoots = _overshoots / 3

    # Return winner of the game
    return _game.get_winner_of_game() ,_actions, _overshoots
