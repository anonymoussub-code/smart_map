import numpy
from copy import deepcopy
class MappingHistory:
    """
    Store only usefull information of a self-play game.
    """
    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.child_visits = []
        self.root_values = []
        self.reanalysed_predicted_root_values = None
        # For PER
        self.priorities = None
        self.game_priority = None
    
    def print_history(self):
        for i in range(len(self.action_history)):
            print(f'State {i}')
            self.observation_history[i].print_state()
            if i< len(self.root_values):
                print('Root Value')
                print(self.root_values[i])
            print()
            if i + 1 < len(self.action_history):
                print('Action | Reward')
                print(self.action_history[i+1],self.reward_history[i+1])
                print()
            if i < len(self.child_visits):
                print('Child Visits')
                print(self.child_visits[i])
                print()
            
        
        print('Reanalysed predicted root values')
        print(self.reanalysed_predicted_root_values)
        print()

        print('Priorities')
        print(self.priorities)
        print()
        
        print('Game Priority')
        print(self.game_priority)
        print()

    def store_search_statistics(self, root, action_space):
        # Turn visit count from root into a policy
        if root is not None:
            sum_visits = sum(child.visit_count for child in root.children.values())
            self.child_visits.append(
                [
                    root.children[a].visit_count / sum_visits
                    if a in root.children
                    else 0
                    for a in action_space
                ]
            )
            self.root_values.append(root.value())
        else:
            self.root_values.append(None)
    def get_stacked_observations(
        self, index, num_stacked_observations, action_space_size
    ):
        """
        Generate a new observation with the observation at the index position
        and num_stacked_observations past observations and actions stacked.
        """
        index = index % len(self.observation_history)

        return self.observation_history[index]
        # # Convert to positive index

        # stacked_observations = deepcopy(self.observation_history[index])
        # for past_observation_index in reversed(
        #     range(index - num_stacked_observations, index)
        # ):
        #     if 0 <= past_observation_index:
        #         previous_observation = numpy.concatenate(
        #             (
        #                 self.observation_history[past_observation_index],
        #                 [
        #                     numpy.ones_like(stacked_observations[0])
        #                     * self.action_history[past_observation_index + 1]
        #                     / action_space_size
        #                 ],
        #             )
        #         )
        #     else:
        #         previous_observation = numpy.concatenate(
        #             (
        #                 numpy.zeros_like(self.observation_history[index]),
        #                 [numpy.zeros_like(stacked_observations[0])],
        #             )
        #         )

        #     stacked_observations = numpy.concatenate(
        #         (stacked_observations, previous_observation)
        #     )

        # return stacked_observations