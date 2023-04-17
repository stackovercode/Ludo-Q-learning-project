import numpy as np
import QLearning
import ludopy
import unittest
import sys
sys.path.append("./LUDOpy/")


def run():
    #  Explore rate: 0.05, discount rate: 0.4 and learning rate: 0.1
    learning_rate_vec = [0.1] #[0.1, 0.2, 0.3, 0.4, 0.5]
    discount_factor_vec = [0.4] #[0.1, 0.2, 0.3, 0.4, 0.5]
    explore_rate_vec = [0.05] #[0.05, 0.10, 0.15, 0.2]

    after = 10

    number_of_runs_without_learning = 25
    number_of_runs_with_learning = 40

    q_player = 0

    size_of_win_rate_vec = (len(explore_rate_vec),len(discount_factor_vec),len(learning_rate_vec), number_of_runs_with_learning)
    win_rate_vec = np.zeros(size_of_win_rate_vec)

    for ER_index, ER_value in enumerate(explore_rate_vec):
        for DF_index, DF_value in enumerate(discount_factor_vec):
            for LR_index, LR_value in enumerate(learning_rate_vec):
                q = QLearning.QLearning(q_player)
                q.training = 1

                q.learning_rate = LR_value
                q.discount_factor = DF_value
                q.explore_rate = ER_value

                for k in range(number_of_runs_with_learning):
                    print('Test:   Number of learning games: ', k, ' ER: ', q.explore_rate, ' DF: ', q.discount_factor, ' LR: ', q.learning_rate)
                    g = ludopy.Game()
                    stop_while = False
                    q.training = 1
                    #number_of_wins_W = 0

                    while not stop_while:
                        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
                         there_is_a_winner), player_i = g.get_observation()

                        if player_i == q_player:
                            piece_to_move = q.update_q_table(player_pieces, enemy_pieces, dice, g, there_is_a_winner)
                            if there_is_a_winner == 1:
                                stop_while = True
                                #number_of_wins_W += 1
                        else:
                            if len(move_pieces):
                                piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
                            else:
                                piece_to_move = -1

                        _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)

                    q.reset_game()
                    if after < k:
                        wins = [0, 0, 0, 0]
                        q.training = 0
                        #number_of_wins_WO = 0

                        number_of_steps = 0
                        for j in range(number_of_runs_without_learning):
                            g = ludopy.Game()
                            stop_while = False
                            while not stop_while:
                                (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
                                 there_is_a_winner), player_i = g.get_observation()
                                if player_i == q_player:
                                    number_of_steps = number_of_steps + 1
                                    piece_to_move = q.update_q_table(player_pieces, enemy_pieces, dice, g, there_is_a_winner)
                                    if there_is_a_winner == 1:
                                        stop_while = True
                                        #number_of_wins_WO += 1
                                else:
                                    if len(move_pieces):
                                        piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
                                    else:
                                        piece_to_move = -1
                                _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)

                            q.reset_game()
                            wins[g.first_winner_was] = wins[g.first_winner_was] + 1
                        win_rate_vec[ER_index][DF_index][LR_index][k] = (wins[q_player] / number_of_runs_without_learning)
                        print('Win rate: ', wins[q_player] / number_of_runs_without_learning)

                q.save_Q_table("Best_learning_parameters" + str(k) + ".npy")

    test_name = "Test_run"
    file_name = test_name + "_data.npy"
    file_ext = file_name.split(".")[-1]
    assert file_ext == "npy", "The file extension has to be npy (numpy file)"
    np.save(file_name, win_rate_vec)

    file_name = test_name + "_parameters.npy"
    file_ext = file_name.split(".")[-1]
    assert file_ext == "npy", "The file extension has to be npy (numpy file)"
    np.save(file_name, [explore_rate_vec, discount_factor_vec, learning_rate_vec, number_of_runs_with_learning, number_of_runs_without_learning])


    return True


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, run())


if __name__ == '__main__':
    unittest.main()

