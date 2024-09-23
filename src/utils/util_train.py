from pathlib import Path
import numpy
from src.utils.util_replay_buffer import UtilReplayBuffer
from src.entities.training_results import TrainingResults
import sys
import numpy as np
from pympler import asizeof
from src.enums.enum_model_name import EnumModelName
class UtilTrain:

    @staticmethod
    def compute_target_value(config, game_history, index):
        # The value target is the discounted root value of the search tree td_steps into the
        # future, plus the discounted sum of all rewards until then.
        bootstrap_index = index + config.td_steps
        if bootstrap_index < len(game_history.root_values):
            root_values = (
                game_history.root_values
                if game_history.reanalysed_predicted_root_values is None
                else game_history.reanalysed_predicted_root_values
            )
            last_step_value = (
                root_values[bootstrap_index]
            )

            value = last_step_value * config.discount**config.td_steps
        else:
            value = 0

        for i, reward in enumerate(
            game_history.reward_history[index + 1 : bootstrap_index + 1]
        ):
            # The value is oriented from the perspective of the current player
            value += (
                reward
            ) * config.discount**i

        return value
    @staticmethod
    def compute_target_value_pre_train(mapping_history, index,last_gae):
        # return mapping_history[index][2] + mapping_history[index+1][3]
        value = 0
        for i, (__,_,reward,v,sum_r) in enumerate(
            mapping_history[index:]
        ):
            # The value is oriented from the perspective of the current player
            value += (
                reward
            ) * 0.997**i
        return value

    @staticmethod
    def make_target_pre_train(mapping_history,gamma=0.99,lamb=0.95):
        """
        Generate targets for every unroll steps.
        """
        states,rewards,values = mapping_history
        last_gae = 0
        len_rollout = len(states)
        adv = [0 for i in range(len_rollout)]
        for t in reversed(range(len_rollout)):
            if t == len_rollout - 1:
                nextvalues = 0
            else:
                nextvalues = values[t+1]

            delta = rewards[t] + gamma * nextvalues - values[t]
            adv[t] = last_gae = delta + gamma * lamb * last_gae
        returns = (numpy.array(adv) + numpy.array(values)).tolist()

        return returns
    @staticmethod
    def make_target(config,game_history,):
        """
        Generate targets for every unroll steps.
        """
        target_values, target_policies, actions = [], [], []
        for current_index in range(config.num_unroll_steps + 1):
            value = UtilTrain.compute_target_value(config,game_history, current_index)

            if current_index < len(game_history.root_values):
                target_values.append(value)
                target_policies.append(game_history.child_visits[current_index])
                actions.append(game_history.action_history[current_index])
            elif current_index == len(game_history.root_values):
                target_values.append(0.)
                # target_rewards.append(game_history.reward_history[current_index])
                # Uniform policy
                target_policies.append(
                    [
                        float('-inf')
                        for _ in range(len(game_history.child_visits[0]))
                    ]
                )
                actions.append(game_history.action_history[current_index])
            else:
                # States past the end of games are treated as absorbing states
                target_values.append(float('-inf'))
                # target_rewards.append(0)
                # Uniform policy
                target_policies.append(
                    [
                        float('-inf')
                        for _ in range(len(game_history.child_visits[0]))
                    ]
                )
                actions.append(numpy.random.choice(config.action_space))
        # return target_values, target_rewards, target_policies, actions
        return target_values, [], target_policies, actions

    @staticmethod
    def collate_fn_decorator(config):
        def collate_fn(samples):
            batch_target_values = []
            batch_target_policies = []
            batch_actions = []
            observation_batch = []
            for     pping in samples:
                target_values, _, target_policies, actions = UtilTrain.make_target(config,mapping)
                batch_actions.append(actions)
                batch_target_policies.append(target_policies)
                batch_target_values.append(target_values)
                observation_batch.append(mapping.observation_history[0])
            return batch_target_values,batch_target_policies,batch_actions,observation_batch
        return collate_fn
    
    @staticmethod
    def collate_fn_decorator_pre_train(config):
        def collate_fn_pre_train(samples):
            batch_states,batch_actions,batch_action_indices,batch_rewards = [],[],[],[]
            batch_values,batch_policy_probs,batch_returns,batch_old_action_probs,batch_old_values = [],[],[],[],[]
            for sample in samples:
                state,action,action_indice,reward,value,policy_prob,ret,old_action_prob,old_value = sample
                batch_states.append(state)
                batch_actions.append(action)
                batch_action_indices.append(action_indice)
                batch_rewards.append(reward)
                batch_values.append(value)
                batch_policy_probs.append(policy_prob)
                batch_returns.append(ret)
                batch_old_action_probs.append(old_action_prob)
                batch_old_values.append(old_value)

            return batch_states,batch_actions,batch_action_indices,batch_rewards,batch_values,batch_policy_probs,batch_returns,batch_old_action_probs,batch_old_values
            # batch_target_values = []
            # batch_actions = []
            # observation_batch = []
            # norm_len_episodes = []
            # batch_action_indices = []
            # batch_sum_rewards = []
            # for mapping in samples:
            #     norm_len_episodes.append(len(mapping)**-1)
            #     target_values, actions = UtilTrain.make_target_pre_train(mapping,config.len_action_space)
            #     np_mapping = numpy.array(mapping)
            #     batch_sum_rewards.append(numpy.sum(np_mapping[:,2]))
            #     batch_actions.append(actions)
            #     if config.model_name == EnumModelName.MAPZERO:
            #         batch_action_indices.append(actions)
            #     else:
            #         action_indices = []
            #         for s,a,r,v,sum_r in mapping[:-1]:
            #             legal_actions = np.array(s.get_legal_actions())
            #             action_indices.append(np.argwhere(legal_actions == a).squeeze().item())
            #         while len(action_indices) < len(actions):
            #             action_indices.append(-1)
            #         batch_action_indices.append(action_indices)


            #     batch_target_values.append(target_values)
            #     observation_batch.append(mapping[0][0])
            # return batch_target_values,batch_actions,observation_batch,norm_len_episodes,batch_action_indices,batch_sum_rewards
        return collate_fn_pre_train
    
    @staticmethod
    def loss_pre_train():
        ...
    
    @staticmethod
    def read_replay_buffers(path,config,type_interconnections):
        training_results = TrainingResults(config.model_name.value,config.arch_dims,type_interconnections.value)
        
        dirr = Path(path)
        buffers = []
        for file in (dirr.rglob('*replay_buffer.pkl')):
            if file.is_file():
                replay_buffer = UtilReplayBuffer.get_replay_buffer(file)
                buffers += list(replay_buffer['buffer'].values())
        print("Total mappings:",len(buffers))
        training_results.total_mapping_samples = len(buffers)
        training_results.total_states = numpy.sum([len(buffer.observation_history) for buffer in buffers])
        total_bytes_size = asizeof.asizeof(buffers)
        total_gb_memory = total_bytes_size / (1024**3)
        training_results.gb_memory = total_gb_memory
        training_results.save_csv()

        return buffers