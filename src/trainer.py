import copy
import time
from src.utils.util_pytorch import UtilPytorch
import numpy
import ray
import torch
import math
@ray.remote
class Trainer:
    """
    Class which run in a dedicated thread to train a neural network and save it
    in the shared storage.
    """

    def __init__(self, model,initial_checkpoint, config):
        self.config = config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Initialize the network
        self.model = model
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if self.config.train_on_gpu else "cpu"))
        self.model.train()

        self.training_step = initial_checkpoint["training_step"]

        if "cuda" not in str(next(self.model.parameters()).device):
            with open("/dev/tty", "w") as terminal:    
                print("You are not training on GPU.\n")

        # Initialize the optimizer
        if self.config.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.lr_init,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.lr_init,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise NotImplementedError(
                f"{self.config.optimizer} is not implemented. You can change the optimizer manually in trainer.py."
            )

        if initial_checkpoint["optimizer_state"] is not None:
            print("Loading optimizer...\n")
            self.optimizer.load_state_dict(
                copy.deepcopy(initial_checkpoint["optimizer_state"])
            )

    def continuous_update_weights(self, replay_buffer, shared_storage):
        # Wait for the replay buffer to be filled
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
            time.sleep(0.1)

        next_batch = replay_buffer.get_batch.remote()
        # Training loop
        while self.training_step < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            index_batch, batch = ray.get(next_batch)
            next_batch = replay_buffer.get_batch.remote()
            self.update_lr()
            (
                priorities,
                total_loss,
                value_loss,
                policy_loss,
            ) = self.update_weights(batch)
            if self.config.PER:
                # Save new priorities in the replay buffer (See https://arxiv.org/abs/1803.00933)
                replay_buffer.update_priorities.remote(priorities, index_batch)

            # Save to the shared storage
            if self.training_step % self.config.checkpoint_interval == 0:
                shared_storage.set_info.remote(
                    {
                        "weights": copy.deepcopy(self.model.get_weights()),
                        "optimizer_state": copy.deepcopy(
                            UtilPytorch.dict_to_cpu(self.optimizer.state_dict())
                        ),
                    }
                )
                if self.config.save_model:
                    shared_storage.save_checkpoint.remote()
            shared_storage.set_info.remote(
                {
                    "training_step": self.training_step,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "total_loss": total_loss,
                    "value_loss": value_loss,
                    "policy_loss": policy_loss,
                }
            )

            # Managing the self-play / training ratio
            if self.config.training_delay:
                time.sleep(self.config.training_delay)
            if self.config.ratio:
                played_games = ray.get(shared_storage.get_info.remote("num_played_games"))
                norm = self.config.adjust_ratio(self.training_step, played_games)
                while (
                   norm*self.training_step
                    / max(
                        1, played_games
                    )
                    > self.config.ratio
                    and self.training_step < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.1)
                    played_games = ray.get(shared_storage.get_info.remote("num_played_games"))
                    norm = self.config.adjust_ratio(self.training_step, played_games)


    def update_weights(self, batch):
        """
        Perform one training step.
        """

        (
            observation_batch,
            action_batch,
            target_value,
            target_reward,
            target_policy,
            weight_batch,
            gradient_scale_batch,
        ) = batch
        # def print_gradient_sizes(model):
        #     for name, param in model.named_parameters():
        #         if param.grad is not None:
        #             print(f"Gradiente do parâmetro {name}: {param.grad.norm().item()}")

        self.optimizer.zero_grad()
        # Keep values as scalars for calculating the priorities for the prioritized replay
        target_value_scalar = numpy.array(target_value, dtype="float32")
        priorities = numpy.zeros_like(target_value_scalar)

        device = next(self.model.parameters()).device
        # if self.config.PER:
            # weight_batch = torch.tensor(weight_batch.copy()).float().to(device)
        # observation_batch = (
        #     torch.tensor(numpy.array(observation_batch)).float().to(device)
        # )   
        action_batch = torch.tensor(action_batch).long().to(device).unsqueeze(-1)
        target_value = torch.tensor(target_value).float().to(device)
        target_reward = torch.tensor(target_reward).float().to(device)
        target_policy = torch.tensor(target_policy).float().to(device)
        gradient_scale_batch = torch.tensor(gradient_scale_batch).float().to(device)
        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1, 1 (unsqueeze)
        # target_value: batch, num_unroll_steps+1
        # target_reward: batch, num_unroll_steps+1
        # target_policy: batch, num_unroll_steps+1, len(action_space)
        # gradient_scale_batch: batch, num_unroll_steps+1

        # target_value: batch, num_unroll_steps+1, 2*support_size+1
        # target_reward: batch, num_unroll_steps+1, 2*support_size+1

        ## Generate predictions
        policy_logits,value, reward, = self.model.initial_inference(
            observation_batch
        )
        curr_state = observation_batch
        predictions = [(value, reward, policy_logits)]
        for i in range(1, action_batch.shape[1]):
            value, reward, policy_logits, curr_state = self.model.recurrent_inference(
                curr_state, action_batch[:, i]
            )
            # Scale the gradient at the start of the dynamics function (See paper appendix Training)
            # hidden_state.register_hook(lambda grad: grad * 0.5)
            predictions.append((value, reward, policy_logits))
        # predictions: num_unroll_steps+1, 3, batch, 2*support_size+1 | 2*support_size+1 | 9 (according to the 2nd dim)
        ## Compute losses
        value_loss, policy_loss = (0, 0)
        value, reward, policy_logits = predictions[0]
        # Ignore reward loss for the first batch step
        current_value_loss, current_policy_loss = self.loss_function(
            value.squeeze(-1),
            policy_logits,
            target_value[:, 0],
            target_policy[:, 0],
        )
        value_loss += current_value_loss
        policy_loss += current_policy_loss
        # Compute priorities for the prioritized replay (See paper appendix Training)
        priorities[:, 0] = (
            numpy.abs(value[:,0].detach().cpu().numpy() - target_value_scalar[:, 0])
            ** self.config.PER_alpha
        )
        for i in range(1, len(predictions)):
            value, reward, policy_logits = predictions[i]
            (
                current_value_loss,
                current_policy_loss,
            ) = self.loss_function(
                value.squeeze(-1),
                policy_logits,
                target_value[:, i],
                target_policy[:, i],
            )

            # Scale gradient by the number of unroll steps (See paper appendix Training)
            # current_value_loss.register_hook(
            #     lambda grad: grad / gradient_scale_batch[:, i]
            # )
            # current_policy_loss.register_hook(
            #     lambda grad: grad / gradient_scale_batch[:, i]
            # )

            value_loss += current_value_loss

            policy_loss += current_policy_loss

            priorities[:, i] = (
                numpy.abs(value[:,0].detach().cpu().numpy() - target_value_scalar[:, i])
                ** self.config.PER_alpha
            )

        policy_loss = policy_loss.mean() / self.config.max_moves
        value_loss = value_loss.mean() * self.config.value_loss_weight / self.config.max_moves
        loss = value_loss + policy_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=1)
        self.optimizer.step()
        self.training_step += 1

        return (
            priorities,
            loss.item(),
            value_loss.item(),
            policy_loss.item(),
        )

    def update_lr(self):
        """
        Update learning rate
        """
        lr = self.config.lr_init * self.config.lr_decay_rate ** (
            self.training_step / self.config.lr_decay_steps
        )
        # print(lr) 
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @staticmethod
    def loss_function(
        value,
        policy_logits,
        target_value,
        target_policy,
    ):
        value = value.unsqueeze(-1)
        target_value = target_value.unsqueeze(-1)
        device = target_value.device
        #fixme: sometimes device goes to cpu  for GeneralMapzero 
        #temp_solution:
        policy_logits = policy_logits.to(device)
        value = value.to(device)
        
        mask_value = target_value == float('-inf')
        masked_value = torch.where(mask_value,0.,value)
        masked_target_value = torch.where(mask_value,0.,target_value)

        value_loss = ((masked_value-masked_target_value)**2)


        mask_policy =( policy_logits == -torch.inf).to(device)  | (target_policy == float('-inf')).to(device)
        masked_target_policy = torch.where(mask_policy,0.,target_policy)
        
        policy_logits = torch.nn.functional.log_softmax(policy_logits,dim=-1)
        masked_policy = torch.where(mask_policy,0.,policy_logits)
        policy_loss = (-masked_target_policy * masked_policy).sum(-1,keepdim=True)

        # print(masked_policy,masked_target_policy)
        # value_loss = torch.zeros((value.size(0),1)).to(device)

        # for i in range(value.size(0)):
        #     cur_mask = mask_value[i].cpu()
        #     masked_target_value = target_value[i][cur_mask]
        #     masked_value =  value[i][cur_mask]
        #     if masked_value.numel() != 0:
        #         assert masked_target_value.numel() != 0
        #         value_loss[i][cur_mask] += ((masked_value-masked_target_value)**2).sum(-1)
        

        # policy_loss = torch.zeros((policy_logits.size(0),1)).to(device)

        # for i in range(value.size(0)):
        #     cur_mask = mask_policy[i].cpu()
        #     masked_target_policy = target_policy[i][cur_mask]
        #     masked_policy =  policy_logits[i][cur_mask]
        #     if masked_policy.numel() != 0:
        #         assert masked_target_policy.numel() != 0
        #         policy_loss[i] += (-masked_target_policy * torch.nn.functional.log_softmax(masked_policy,dim=-1)).sum(-1)
        return value_loss, policy_loss

