import numpy as np
import game_env
from game_env import Game2048Env
import pytest

class Test2048Logic():
    def test_combine(self):
        game = Game2048Env()
        # Test not combining
        assert game.combine([0, 0, 0, 0]) == ([0, 0, 0, 0], 0)
        assert game.combine([2, 4, 8, 16]) == ([2, 4, 8, 16], 0)

        # Test combining
        # Left same same
        assert game.combine([2, 2, 8, 0]) == ([4, 8, 0, 0], 4)
        # Middle the same
        assert game.combine([4, 2, 2, 4]) == ([4, 4, 4, 0], 4)
        # Left and middle the same
        assert game.combine([2, 2, 2, 8]) == ([4, 2, 8, 0], 4)
        # Right the same
        assert game.combine([2, 8, 4, 4]) == ([2, 8, 8, 0], 8)
        # Left and right the same
        assert game.combine([2, 2, 4, 4]) == ([4, 8, 0, 0], 12)
        # Right and middle the same
        assert game.combine([2, 4, 4, 4]) == ([2, 8, 4, 0], 8)
        # All the same
        assert game.combine([4, 4, 4, 4]) == ([8, 8, 0, 0], 16)

        # Test short input
        assert game.combine([]) == ([0, 0, 0, 0], 0)
        assert game.combine([0]) == ([0, 0, 0, 0], 0)
        assert game.combine([2]) == ([2, 0, 0, 0], 0)
        assert game.combine([2, 4]) == ([2, 4, 0, 0], 0)
        assert game.combine([2, 2, 8]) == ([4, 8, 0, 0], 4)

    def test_shift(self):
        game = Game2048Env()
        # Shift left without combining
        assert game.shift([0, 0, 0, 0], 0) == ([0, 0, 0, 0], 0)
        assert game.shift([0, 2, 0, 0], 0) == ([2, 0, 0, 0], 0)
        assert game.shift([0, 2, 0, 4], 0) == ([2, 4, 0, 0], 0)
        assert game.shift([2, 4, 8, 16], 0) == ([2, 4, 8, 16], 0)

        # Shift left and combine
        assert game.shift([0, 2, 2, 8], 0) == ([4, 8, 0, 0], 4)
        assert game.shift([2, 2, 2, 8], 0) == ([4, 2, 8, 0], 4)
        assert game.shift([2, 2, 4, 4], 0) == ([4, 8, 0, 0], 12)

        # Shift right without combining
        assert game.shift([0, 0, 0, 0], 1) == ([0, 0, 0, 0], 0)
        assert game.shift([0, 2, 0, 0], 1) == ([0, 0, 0, 2], 0)
        assert game.shift([0, 2, 0, 4], 1) == ([0, 0, 2, 4], 0)
        assert game.shift([2, 4, 8, 16], 1) == ([2, 4, 8, 16], 0)

        # Shift right and combine
        assert game.shift([2, 2, 8, 0], 1) == ([0, 0, 4, 8], 4)
        assert game.shift([2, 2, 2, 8], 1) == ([0, 2, 4, 8], 4)
        assert game.shift([2, 2, 4, 4], 1) == ([0, 0, 4, 8], 12)

    def test_move(self):
        # Test a bunch of lines all moving at once.
        game = Game2048Env()
        # Test shift up
        game.set_board(np.array([
            [0, 2, 0, 4],
            [2, 2, 8, 0],
            [2, 2, 2, 8],
            [2, 2, 4, 4]]))
        assert game.move(0) == 12
        assert np.array_equal(game.get_board(), np.array([
            [4, 4, 8, 4],
            [2, 4, 2, 8],
            [0, 0, 4, 4],
            [0, 0, 0, 0]]))
        # Test shift right
        game.set_board(np.array([
            [0, 2, 0, 4],
            [2, 2, 8, 0],
            [2, 2, 2, 8],
            [2, 2, 4, 4]]))
        assert game.move(1) == 20
        assert np.array_equal(game.get_board(), np.array([
            [0, 0, 2, 4],
            [0, 0, 4, 8],
            [0, 2, 4, 8],
            [0, 0, 4, 8]]))
        # Test shift down
        game.set_board(np.array([
            [0, 2, 0, 4],
            [2, 2, 8, 0],
            [2, 2, 2, 8],
            [2, 2, 4, 4]]))
        assert game.move(2) == 12
        assert np.array_equal(game.get_board(), np.array([
            [0, 0, 0, 0],
            [0, 0, 8, 4],
            [2, 4, 2, 8],
            [4, 4, 4, 4]]))
        # Test shift left
        game.set_board(np.array([
            [0, 2, 0, 4],
            [2, 2, 8, 0],
            [2, 2, 2, 8],
            [2, 2, 4, 4]]))
        assert game.move(3) == 20
        assert np.array_equal(game.get_board(), np.array([
            [2, 4, 0, 0],
            [4, 8, 0, 0],
            [4, 2, 8, 0],
            [4, 8, 0, 0]]))

        # Test that doing the same move again (without anything added) is illegal
        with pytest.raises(game_env.IllegalMove):
            game.move(3)

        # Test a follow on move from the first one
        assert game.move(2) == 8 # shift down
        assert np.array_equal(game.get_board(), np.array([
            [0, 4, 0, 0],
            [2, 8, 0, 0],
            [4, 2, 0, 0],
            [8, 8, 8, 0]]))

    def test_highest(self):
        game = Game2048Env()
        game.set_board(np.array([
            [0, 2, 0, 4],
            [2, 2, 8, 0],
            [2, 2, 2048, 8],
            [2, 2, 4, 4]]))
        assert game.highest() == 2048

    def test_isend(self):
        game = Game2048Env()
        game.set_board(np.array([
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2]]))
        assert game.isend() == False
        game.set_board(np.array([
            [2, 4, 8, 16],
            [4, 8, 16, 2],
            [8, 16, 2, 4],
            [16, 2, 4, 8]]))
        assert game.isend() == True


# if __name__ == "__main__":
#     # play with 2048Game
#     # import pdb; pdb.set_trace()
#     game = Game2048Env()
#     observation, reward, done, info = game.step(2)
#     print(observation)
#     # game.render()
#     game.step(1)
#     # game.render()
#     game.step(2)
#     # game.render()
#     game.step(1)
#     # game.render()
#     game.close()
