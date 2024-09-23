from pathlib import Path 
from src.utils.util_replay_buffer import UtilReplayBuffer
from src.mapping_dataset import MappingDataset
from torch.utils.data.dataloader import DataLoader
from src.utils.util_configs import UtilConfigs
from configs.config_mapzero import ConfigMapzero
import numpy
from src.enums.enum_interconnections import EnumInterconnections
from src.enums.enum_mode import EnumMode
import torch
from src.graphs.graphs.networkx_graph import NetworkXGraph
from torch.utils.tensorboard import SummaryWriter
from src.trainer import Trainer
from src.utils.util_dfg import UtilDFG
from src.utils.util_initialize import UtilInitialize
from src.mapping_history import MappingHistory
from src.mcts.mcts import MCTS
from tqdm import tqdm
from src.utils.util_train import UtilTrain
from src.utils.util_eval import UtilEval
import sys
from src.utils.util_module import UtilModule
import time
from src.entities.training_results import TrainingResults
from configs.config_yoto_mapzero import ConfigYOTOMapzero
from configs.config_yott_mapzero import ConfigYOTTMapzero
import os
from torch.utils.data import Dataset, DataLoader, random_split
# @torch.no_grad
# def eval_model_fn(config,model,states,enviroment):
#     model.eval()
#     results = []
#     for state in states:
#         results.append(UtilEval.play_game(config,model,state,enviroment,config.num_max_expansion_test)[0])
#     mean_routing_penalty = numpy.mean([sum(history.reward_history) for history in results])
#     vmr = numpy.sum([1 if history.observation_history[-1].mapping_is_valid else 0 for history in results])/len(states)
#     mean_reward = numpy.mean([numpy.mean(history.reward_history) for history in results])
#     return mean_routing_penalty,vmr, mean_reward

@torch.no_grad
def eval_model_fn(config,model,dataloader):
    model.eval()
    epoch_value_loss = 0
    epoch_policy_loss = 0
    for target_value,target_policy,action_batch,observation_batch in dataloader:
        device = next(model.parameters()).device
        action_batch = torch.tensor(action_batch).long().to(device).unsqueeze(-1)
        target_value = torch.tensor(target_value).float().to(device)
        target_policy = torch.tensor(target_policy).float().to(device)
    
        policy_logits,value, reward, = model.initial_inference(
            observation_batch
        )
        curr_state = observation_batch
        predictions = [(value, reward, policy_logits)]
        for i in range(1, action_batch.shape[1]):
            value, reward, policy_logits, curr_state = model.recurrent_inference(
                curr_state, action_batch[:, i]
            )

            predictions.append((value, reward, policy_logits))

        value_loss, policy_loss = (0, 0)
        value, reward, policy_logits = predictions[0]

        current_value_loss, current_policy_loss = Trainer.loss_function(
            value.squeeze(-1),
            policy_logits,
            target_value[:, 0],
            target_policy[:, 0],
        )
        value_loss += current_value_loss
        policy_loss += current_policy_loss


        for i in range(1, len(predictions)):
            value, reward, policy_logits = predictions[i]
            (
                current_value_loss,
                current_policy_loss,
            ) = Trainer.loss_function(
                value.squeeze(-1),
                policy_logits,
                target_value[:, i],
                target_policy[:, i],
            )
            value_loss += current_value_loss

            policy_loss += current_policy_loss

        policy_loss = policy_loss.mean() / config.max_moves
        value_loss = value_loss.mean() / config.max_moves

        epoch_value_loss += value_loss.item()
        epoch_policy_loss += policy_loss.item()
    return epoch_value_loss + epoch_policy_loss, epoch_value_loss,epoch_policy_loss

def train(model_config,batch_size,eval_model=False):
    config = model_config.get_config()
    path_to_ckpt_model = model_config.get_path_to_ckpt_model()
    if os.path.exists(path_to_ckpt_model):
        print()
        print(f'Model {config.model_name.value} from config {model_config.__class__.__name__} has already been trained. Please delete or move the file {path_to_ckpt_model} if you want to train again.')
        print()
        return None

    if config.seed:
        generator = torch.manual_seed(config.seed)
        numpy.random.seed(config.seed)


    if not eval_model:
        print("\nModel will not be evaluated during training as `eval_model` is set to False.\n")

    collate_fn = UtilTrain.collate_fn_decorator(config)
    mappings = UtilTrain.read_replay_buffers(config.results_path,config,model_config.type_interconnnections)
    mapping_dataset = MappingDataset(mappings)
    if config.seed:
        train_dataset, test_dataset = random_split(mapping_dataset,[0.9,0.1],generator=generator)
    else:
        train_dataset, test_dataset = random_split(mapping_dataset,[0.9,0.1])

    model = config.class_model(*config.model_instance_args)
    model = model.to(config.device)
    train_dataloader = DataLoader(train_dataset,batch_size,shuffle=True,num_workers=4,collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset,batch_size,shuffle=True,num_workers=4,collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,max(1,int(len(train_dataloader)*0.4)),0.99)
    # if eval_model:
    #     eval_states = UtilEval.get_initial_eval_states(config,model_config.get_cgra())

    writer = SummaryWriter(config.results_path)

    print(
        f"\nTraining on {config.device}...\nRun tensorboard --logdir {config.results_path} and go to http://localhost:6006/ to see in real time the training performance.\n"
    )

    hp_table = [
        f"| {key} | {value} |" for key, value in config.__dict__.items()
    ]
    writer.add_text(
        "Hyperparameters",
        "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table),
    )

    writer.add_text(
        "Model summary",
        str(model).replace("\n", " \n\n"),
    )

    total_train_time = 0
    total_eval_time = 0
    min_loss = float('inf')
    for epoch in range(config.epochs):
        init_train_time = time.time()
        epoch_value_loss = 0
        epoch_policy_loss = 0
        model.train()
        for target_value,target_policy,action_batch,observation_batch in train_dataloader:
            optimizer.zero_grad()
            device = next(model.parameters()).device

            action_batch = torch.tensor(action_batch).long().to(device).unsqueeze(-1)
            target_value = torch.tensor(target_value).float().to(device)
            target_policy = torch.tensor(target_policy).float().to(device)
        
            policy_logits,value, reward, = model.initial_inference(
                observation_batch
            )
            curr_state = observation_batch
            predictions = [(value, reward, policy_logits)]
            for i in range(1, action_batch.shape[1]):
                value, reward, policy_logits, curr_state = model.recurrent_inference(
                    curr_state, action_batch[:, i]
                )

                predictions.append((value, reward, policy_logits))

            value_loss, policy_loss = (0, 0)
            value, reward, policy_logits = predictions[0]

            current_value_loss, current_policy_loss = Trainer.loss_function(
                value.squeeze(-1),
                policy_logits,
                target_value[:, 0],
                target_policy[:, 0],
            )
            value_loss += current_value_loss
            policy_loss += current_policy_loss


            for i in range(1, len(predictions)):
                value, reward, policy_logits = predictions[i]
                (
                    current_value_loss,
                    current_policy_loss,
                ) = Trainer.loss_function(
                    value.squeeze(-1),
                    policy_logits,
                    target_value[:, i],
                    target_policy[:, i],
                )
                value_loss += current_value_loss

                policy_loss += current_policy_loss

            policy_loss = policy_loss.mean() / config.max_moves
            value_loss = value_loss.mean() / config.max_moves
            loss = value_loss + policy_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=1)
            optimizer.step()
            scheduler.step()
            epoch_value_loss += value_loss.item()
            epoch_policy_loss += policy_loss.item()

        epoch_policy_loss/=len(train_dataloader)
        epoch_value_loss/=len(train_dataloader)
        lr = optimizer.param_groups[0]['lr']
        total_loss = epoch_policy_loss + epoch_value_loss


        final_train_time = time.time()
        total_train_time += final_train_time - init_train_time

        writer.add_scalar("1.Loss/1.Train_Total_Loss",total_loss,epoch)
        writer.add_scalar("1.Loss/2.Train_Policy_Loss",epoch_policy_loss,epoch)
        writer.add_scalar("1.Loss/3.Train_Value_Loss",epoch_value_loss,epoch)
        if eval_model:
            init_eval_time = time.time()
            eval_loss,eval_loss_value,eval_policy_loss = eval_model_fn(config,model,test_dataloader)
            final_eval_time = time.time()
            total_eval_time += final_eval_time - init_eval_time
            writer.add_scalar("1.Loss/4.Eval_Total_Loss",eval_loss,epoch)
            writer.add_scalar("1.Loss/5.Eval_Policy_Loss",eval_loss_value,epoch)
            writer.add_scalar("1.Loss/6.Eval_Value_Loss",eval_policy_loss,epoch)

        writer.add_scalar("3.Control/1.Learning_Rate",lr,epoch)
        print(f'Epoch: {epoch + 1}. Train: Total Loss: {total_loss:.5f}. Value Loss: {epoch_value_loss:.5f}. Policy Loss: {epoch_policy_loss:.5f} | Eval: Total Loss: {eval_loss:.5f}. Value Loss: {eval_loss_value:.3f}. Policy Loss: {eval_policy_loss} | lr: {lr:.6f}. Time: {final_train_time-init_train_time:.3f}')

        if total_loss < min_loss:
            min_loss = total_loss
            print('Model saved.')
            torch.save(model.state_dict(),path_to_ckpt_model)
    
    print()
    print(f'Model saved in {path_to_ckpt_model}')
    print()
    train_results = TrainingResults(config.model_name.value,config.arch_dims,model_config.type_interconnnections.value)
    train_results.training_time = total_train_time
    train_results.eval_time = total_eval_time
    train_results.save_csv()

if __name__ == "__main__":
    args = sys.argv
    config_module = args[1]
    config_class = args[2]
    interconnection_style = args[3]
    interconnection_style = EnumInterconnections(interconnection_style)
    arch_dims = tuple(map(int,args[4].split('_')))
    mode = EnumMode(args[5])
    eval_model = True if arch_dims == (4,4) else False
    model_config = UtilModule.load_class_from_file(config_module,config_class,interconnection_style,arch_dims,mode)
    
    # interconnection_style = EnumInterconnections.ONE_HOP
    # arch_dims = (4,4)
    # config_class = 'test'
    # model_config = ConfigMapzero(EnumInterconnections.OH_TOR_DIAG,(8,8),EnumMode.DATA_GENERATION)
    # model_config = ConfigMapzero(EnumInterconnections.OH_TOR_DIAG,(4,4),EnumMode.TRAIN)
    # model_config = ConfigYOTTMapzero(EnumInterconnections.ONE_HOP,(4,4),EnumMode.TRAIN)
    # model_config = ConfigYOTOMapzero(EnumInterconnections.ONE_HOP,(4,4),EnumMode.TRAIN)
    
    batch_size = 32
    train(model_config,batch_size,eval_model)