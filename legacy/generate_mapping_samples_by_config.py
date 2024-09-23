from src.graphs.graph_interface import GraphInterface
from src.graphs.dfgs.dfg_mapzero import DFGMapZero
from src.graphs.graphs.networkx_graph import NetworkXGraph
from src.utils.util_dfg import UtilDFG
from src.graphs.cgras.cgra_mapzero import CGRAMapzero
from src.enums.enum_functionalitie_pe import EnumFunctionalitiePE
from src.utils.util_interconnections import UtilInterconnections
from src.enums.enum_interconnect_style import EnumInterconnectStyle
from src.models.mapzero import MapZero
from src.rl.enviroments.mapzero_enviroment import MapZeroEnviroment
from src.replay_buffer import ReplayBuffer
from src.shared_storage import SharedStorage
from src.reanalyse import Reanalyse
import logging
from src.rl.states.mapping_state_mapzero import MappingStateMapZero
from src.mcts.mcts import MCTS
import torch
from torch import nn
from src.config import Config
from src.trainer import Trainer
import ray
from src.self_play import SelfPlay
import time
from src.utils.softmax_temperature import SoftmaxTemperature
from model_launcher import ModelLauncher
from src.utils.util_checkpoint import UtilCheckpoint
from src.utils.util_replay_buffer import UtilReplayBuffer
from pathlib import Path
import os
from src.entities.training_results import TrainingResults
from src.utils.util_path import UtilPath
from src.enums.enum_interconnections import EnumInterconnections
import sys
from src.utils.util_module import UtilModule
from src.utils.util_initialize import UtilInitialize
from src.utils.util_configs import UtilConfigs
from configs.config_mapzero import ConfigMapzero
from configs.config_yoto_mapzero import ConfigYOTOMapzero
from configs.config_yott_mapzero import ConfigYOTTMapzero
import numpy as np
from src.enums.enum_mode import EnumMode

def generate_dataset(config:Config, dfg_name,path_to_dot_file, cgra, weights):
    if config.seed:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
    dfg_graph : GraphInterface = NetworkXGraph(path_to_dot_file)
    assert dfg_graph.num_nodes() <= cgra.len_vertices()
    node_to_operation = UtilDFG.generate_node_to_operation_by_networkx_graph(dfg_graph.graph,"opcode")
    dfg = UtilInitialize.initialize_dfg_from_class_name(config.dfg_class_name,dfg_graph,node_to_operation)

    checkpoint = UtilCheckpoint.get_checkpoint(
    )

    model = config.class_model(*config.model_instance_args)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    if weights is None:  
        checkpoint['weights'] = model.get_weights()
    else:
        checkpoint['weights'] = weights
    initial_state = UtilConfigs.get_mapping_state_by_model_name(config.model_name,dfg,cgra,dfg.get_next_node_to_be_mapped(),
                                            config.distance_func)

    launcher = ModelLauncher(model,dfg_name, config,initial_state,config.enviroment,checkpoint,None)

    launcher.train()
    return launcher.checkpoint['weights']

if __name__ == "__main__":
    args = sys.argv
    config_module = args[1]
    config_class = args[2]
    interconnection_style = args[3]
    interconnection_style = EnumInterconnections(interconnection_style)
    arch_dims = tuple(map(int,args[4].split('_')))
    mode = EnumMode(args[5])
    model_config = UtilModule.load_class_from_file(config_module,config_class,interconnection_style,arch_dims,mode)
    # arch_dims = (8,8)
    # interconnection_style = EnumInterconnections.OH_TOR_DIAG
    # config_class = 'test'
    # model_config = ConfigMapzero(EnumInterconnections.OH_TOR_DIAG,(4,4),EnumMode.DATA_GENERATION)
    cgra = model_config.get_cgra()
    config = model_config.get_config()
    dirs = map(str,config.dataset_range)
    training_results = TrainingResults(config.model_name.value,arch_dims,interconnection_style.value)
    init_time = time.time()
    weights = None
    for cur_dir in dirs:
        path = Path(f"{config.path_to_train_dataset}{cur_dir}{os.sep}")
        files = [str(file) for file in path.rglob('*.dot') if file.is_file()]
        training_steps = UtilConfigs.get_training_steps(cur_dir,model_config.get_type_interconnnections(),config.arch_dims)
        config.set_training_steps(training_steps)
        for file in (files):
            dirname,filename = UtilPath.get_last_two_folders(file)
            dfg_name = f"{dirname}{os.sep}{filename.replace('.dot','')}"
            if os.path.exists(config.results_path + dfg_name + os.sep + 'replay_buffer.pkl'):
                print()
                print(f'Samples already generated for {dfg_name}. \nTo collect the correct total execution time, delete the buffers for config {config_class}.\n')
                continue
            print()
            print(f'Training {dfg_name}\n')
            weights = generate_dataset(config,dfg_name,file,cgra,weights)
    final_time = time.time()
    training_results.sample_generation_time = final_time - init_time
    training_results.save_csv()


        
                