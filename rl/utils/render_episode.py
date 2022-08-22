import copy

class RenderEpisode:
    def __init__(self, env):
        self.env = env
        self.episode_state = []
        self.episode_action = []

    def append_episode_state(self):
        state = self.env.get_board()
        # print(state)
        # self.episode_state.append(state)  # bug: state is a list, need to deepcopy
        self.episode_state.append(copy.deepcopy(state))
    
    def append_episode_action(self, action):
        self.episode_action.append(action)

    def save_episode(self, episode_num):
        file_name = 'episode_{}'.format(episode_num)
        with open(file_name, 'w') as f:
            for state in self.episode_state:
                f.write(state)

        self.reset_episode()            


    def show_episode(self):
        for state in self.episode_state:
            print(state)
            print()
            # print('select action:', )

        self.reset_episode()
        

    def reset_episode(self):
        self.episode_state = []
        self.episode_action = []
