"""Preprocessors for Atari pixel output."""

import numpy as np
from PIL import Image

class HistoryPreprocessor:
    """Keeps the last k states.
    Useful for domains where you need velocities, but the state
    contains only positions.
    When the environment starts, this will just fill the initial
    sequence values with zeros k times.
    Parameters
    ----------
    history_length: int
      Number of previous states to prepend to state being processed.
    """

    def __init__(self, input_shape, history_length=1):
        self._WIDTH = input_shape[0]
        self._HEIGHT = input_shape[1]
        self.history = np.zeros(shape=(1, self._WIDTH, self._HEIGHT, history_length+1), dtype = np.uint8)
        self.history_length = history_length


    def process_state_for_memory(self, state):
        """You only want history when you're deciding the current action to take.
        Returns last history_length processed states, where each is the max of
        the raw state and the previous raw state.
        """
        self.history[0,:,:,1:]=self.history[0,:,:,:self.history_length]
        self.history[0,:,:,0]=state

    def reset(self):
        """Reset the history sequence.
        Useful when you start a new episode.
        """
        self.history = np.zeros(shape=(1, self._WIDTH, self._HEIGHT, self.history_length+1), dtype = np.uint8)

    def get_state(self):
        result = np.zeros(shape=(1, self._WIDTH, self._HEIGHT, self.history_length), dtype = np.uint8)

        for i in range(self.history_length):
            result[0,:,:,i]=np.maximum(self.history[0,:,:,i], self.history[0,:,:,i+1])
        return result



class AtariPreprocessor:
    """Converts images to greyscale and downscales.
    Based on the preprocessing step described in:
    @article{mnih15_human_level_contr_throug_deep_reinf_learn,
    author =	 {Volodymyr Mnih and Koray Kavukcuoglu and David
                  Silver and Andrei A. Rusu and Joel Veness and Marc
                  G. Bellemare and Alex Graves and Martin Riedmiller
                  and Andreas K. Fidjeland and Georg Ostrovski and
                  Stig Petersen and Charles Beattie and Amir Sadik and
                  Ioannis Antonoglou and Helen King and Dharshan
                  Kumaran and Daan Wierstra and Shane Legg and Demis
                  Hassabis},
    title =	 {Human-Level Control Through Deep Reinforcement
                  Learning},
    journal =	 {Nature},
    volume =	 518,
    number =	 7540,
    pages =	 {529-533},
    year =	 2015,
    doi =        {10.1038/nature14236},
    url =	 {http://dx.doi.org/10.1038/nature14236},
    }
    You may also want to max over frames to remove flickering. Some
    games require this (based on animations and the limited sprite
    drawing capabilities of the original Atari).
    Parameters
    ----------
    new_size: 2 element tuple
      The size that each image in the state should be scaled to. e.g
      (84, 84) will make each image in the output have shape (84, 84).
    """

    def __init__(self, input_shape):
        self._WIDTH = input_shape[0]
        self._HEIGHT = input_shape[1]


    def process_state_for_memory(self, state):
        """Scale, convert to greyscale and store as uint8.
        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eigth of the bytes (depending on float32 or float64)
        """
        image = Image.fromarray(state)
        image = image.convert('L')
        image = image.resize((self._WIDTH, self._HEIGHT), Image.LANCZOS)
        #image.show()
        image = np.array(image,dtype=np.uint8)
        return image.reshape((1, self._WIDTH, self._HEIGHT))


class Preprocessor:
    """Combination of both an Atari preprocessor and history preprocessor."""
    def __init__(self, window_size, input_shape):
        self._atari_preprocessor = AtariPreprocessor(input_shape)
        self._history_preprocessor = HistoryPreprocessor(input_shape, history_length=window_size)

    def process_state_for_memory(self, state):
        if state.shape == (210, 160, 3):
            state = self._atari_preprocessor.process_state_for_memory(state)
            state = self._history_preprocessor.process_state_for_memory(state)
        else:
            raise Exception("Shape Error in preprocessor"+str(state.shape))
        return state

    def reset(self):
        self._history_preprocessor.reset()

    def process_reward(self, reward):
        """Get sign of reward: -1, 0 or 1."""
        return np.sign(reward)

    def get_state(self):
        return self._history_preprocessor.get_state()

    def state2float(self, state):
        if state.dtype != np.uint8:
            raise Exception("Error, State should be in np.unit8")
        return state.astype(np.float16) / 255.0