import ray
@ray.remote(num_cpus=0, num_gpus=0)
class CPUActor:
    # Trick to force DataParallel to stay on CPU to get weights on CPU even if there is a GPU
    def __init__(self):
        pass

    def get_initial_weights(self,model):
        weigths = model.get_weights()
        summary = str(model).replace("\n", " \n\n")
        return weigths, summary