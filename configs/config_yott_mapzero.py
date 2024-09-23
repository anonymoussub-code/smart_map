from configs.general_config_model import GeneralConfigModel
from src.rl.enviroments.yott_mapzero_enviroment import YOTTMapzeroEnviroment
from src.graphs.dfgs.yoto_dfg import YOTODFG
from src.graphs.cgras.cgra_traversal import CGRATraversal
from src.models.general_mapzero import GeneralMapzero
import torch
from src.enums.enum_dfg import EnumDFG
from src.utils.util_configs import UtilConfigs
from src.enums.enum_model_name import EnumModelName
from src.enums.enum_cgra import EnumCGRA
class ConfigYOTTMapzero(GeneralConfigModel):
    def __init__(self,type_interconnections,arch_dims,mode):
        model_name = EnumModelName.YOTT_MAPZERO
        enviroment = YOTTMapzeroEnviroment()
        model_instance_args = [9,7,32,arch_dims[0]*arch_dims[1],torch.float32,enviroment]
        dfg_class_name = EnumDFG.YOTT_DFG
        cgra_class_name = EnumCGRA.CGRA_TRAVERSAL

        super().__init__(type_interconnections,arch_dims,model_name,enviroment,GeneralMapzero,
                            model_instance_args,dfg_class_name,cgra_class_name,mode)
        self.config.distance_func = UtilConfigs.get_distance_func_by_type_interconnections(type_interconnections)
