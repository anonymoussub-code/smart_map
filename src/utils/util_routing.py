from src.utils.util_calculations import UtilCalculations
import heapq
class UtilRouting:   
    """
    Class to perform routing.
    """
    @staticmethod
    def route(father_pe,arch_dims, child_pe,used_pes,pes_to_routing, free_interconnections,out_vertices):
        """
        Performs routing between two PEs using BFS.
        Args:
            father_pe (int or tuple): Source PE to perform routing.
            child_pe (int or tuple): Destination PE to perform routing.
            used_pes (set): PEs already placed.
            pes_to_routing (dict[tuple,list]): Dict (k,v) where k is two PEs (source and destination, respectively) and v is a list containing 
                                               the used edges for routing between this PEs.
            free_interconnnections (dict): Dict (k,v) where k is a PE and v is a list of PEs indicating that there exist directly free edges
                                           (interconnections) between them.
            out_vertices (dict): Dict (k,v) where k is a PE and v is a list of PEs indicating that they are connected.
        Returns:
            dict: Updated pes_to_routing with new routing, if there was any.
            dict: Updated free_interconnections according to the new routing.
            int: Number of interconnections used.
        Tested:
            False                
        """

        path = UtilRouting.a_star_routing(father_pe, arch_dims,child_pe, used_pes, out_vertices,free_interconnections)
        pes_to_routing[(father_pe,child_pe)] = path
        for i in range(len(path) - 1):
            free_interconnections[path[i]].remove(path[i + 1])
        cost = len(pes_to_routing[(father_pe,child_pe)])
        return pes_to_routing,free_interconnections,cost
    
    @staticmethod
    def a_star_routing(init_state, arch_dims, final_state,used_pes, out_vertexes: dict,free_interconnections):
        pe_pos = lambda id,row,col: (id // row ,id % col) 
        calc_dist_fn = UtilCalculations.calc_dist_manhattan
        
        heap = []
        heapq.heappush(heap,(0,init_state))

        visited = {init_state:True}
        
        for pe in used_pes:
            if pe != final_state:
                visited[pe] = True
        father = {}

        while len(heap) > 0:
            _,curr_node =heapq.heappop(heap) 

            if curr_node == final_state:
                aux = final_state
                path = []
                while aux != init_state:
                    path.append(aux)
                    aux = father[aux]
                path.append(aux)
                path.reverse()
                return path

            for neighboor in out_vertexes:
                if neighboor not in visited and neighboor in free_interconnections[curr_node]:
                    heapq.heappush(heap,(calc_dist_fn(pe_pos(curr_node,arch_dims[0],arch_dims[1]),pe_pos(neighboor,arch_dims[0],arch_dims[1])),neighboor))
                    visited[neighboor] = True
                    father[neighboor] = curr_node
        return []

    

    
    @staticmethod
    def bfs_routing(init_state, final_state,used_pes, out_vertexes: dict,free_interconnections):
        """
        BFS code to perform the routing.

        Args:
            init_state (int or tuple): Source node to perform the search.
            final_state (int or tuple): Destination node to perform the search.
            used_pes (set): PEs already placed.
            out_vertexes (dict): Dict (k,v) where k is a PE and v is a list of PEs indicating that they are connected.
            free_interconnections (dict): Dict (k,v) where k is a PE and v is a list of PEs indicating that there exist directly free edges
                                   (interconnections) between them.
        Returns:
            list: Path between init_state and final_state, if there was any.
        Tested:
            False
        """
        fifo = []
        fifo.append(init_state)

        visited = {init_state:True}
        
        for pe in used_pes:
            if pe != final_state:
                visited[pe] = True
        father = {}

        while len(fifo) > 0:
            curr_node = fifo.pop(0)

            if curr_node == final_state:
                aux = final_state
                path = []
                while aux != init_state:
                    path.append(aux)
                    aux = father[aux]
                path.append(aux)
                path.reverse()
                return path

            for neighboor in out_vertexes:
                if neighboor not in visited and neighboor in free_interconnections[curr_node]:
                    fifo.append(neighboor)
                    visited[neighboor] = True
                    father[neighboor] = curr_node
        return []

    

