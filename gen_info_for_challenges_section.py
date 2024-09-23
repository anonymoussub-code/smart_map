from math import factorial
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from src.graphs.cgras.cgra import CGRA
from src.graphs.graphs.networkx_graph import NetworkXGraph
from src.utils.util_interconnections import UtilInterconnections
import itertools
from src.utils.util_routing import UtilRouting
from src.utils.alap import ALAP
import copy
from src.utils.util_mapping import UtilMapping
from decimal import Decimal, getcontext
import seaborn as sns
import pandas as pd

def generate_num_placement_possibilities():
    def shorten_large_number(num):
        magnitude = 0
        num = Decimal(num) 
        
        while abs(num) >= 1000:
            magnitude += 1
            num /= Decimal(1000.0)
        return f"{num:.2f}E+{magnitude * 3}"
    
    dims = (4, 8, 16)
    for dim in dims:
        num_placements = factorial(dim * dim)
        print(f'{dim}x{dim}: {shorten_large_number(num_placements)}')


def generate_valid_mapping_rates_3x3():
    df = pd.DataFrame({'Interconnection Style':[],'dfg_name':[],'valid_sol_rate':[]})
    def unique_permutations(lst):
        sett = set()
        for perm in itertools.permutations(lst):
            if perm not in sett:
                sett.add(perm)
                yield perm
    def route(node,pes_to_routing,free_interconnections,in_vertices_dfg,node_to_pe,out_vertices_cgra):
        used_pes=range(0,4)
        cost = None
        pe = node_to_pe[node]
        for father in in_vertices_dfg[node]:
            father_pe = node_to_pe[father]
            pes_to_routing,free_interconnections,cost = UtilRouting.route(father_pe,dim_arch,pe,used_pes ,pes_to_routing,free_interconnections,out_vertices_cgra)
        return cost,pes_to_routing,free_interconnections
    dim_arch = (3,3)
    sns.set(style="whitegrid", palette="pastel", font_scale=1.2)
    archs =[
        "OH+Tor+Diag",
        "One Hop",
        "Mesh"]
    cgras = [
        CGRA(NetworkXGraph(),dim_arch,dim_arch,None,None,-1,UtilInterconnections.generate_one_hop_interconnection_by_pe_pos,
                                                            UtilInterconnections.generate_toroidal_interconnection_by_pe_pos,
                                                            UtilInterconnections.generate_diagonal_interconnection_by_pe_pos),
            CGRA(NetworkXGraph(),dim_arch,dim_arch,None,None,-1,UtilInterconnections.generate_one_hop_interconnection_by_pe_pos),
            CGRA(NetworkXGraph(),dim_arch,dim_arch,None,None,-1,UtilInterconnections.generate_mesh_interconnection_by_pe_pos)]
    files = ["4/V_4_E_4","5/V_5_E_5","6/V_6_E_5","7/V_7_E_8","8/V_8_E_8","9/V_9_E_11"]
    num_placements = factorial(dim_arch[0]*dim_arch[1])
    for file in files:
        print(file)
        graph = NetworkXGraph(f"benchmarks/synthetics/{file}.dot")
        vertices = list(graph.get_vertices())
        edges = graph.get_edges()
        alap_values = ALAP.get_alap_values(vertices,edges)
        cp_vertices = copy.deepcopy(vertices)
        while len(vertices)<dim_arch[0]*dim_arch[1]:
            vertices.append(-1)
        order = copy.copy(alap_values)
        order = sorted(order.items(), key= lambda item: item[1])
        order = [k for k,v in order]
        in_vertices = graph.calc_in_vertex()
        out_vertices = graph.calc_out_vertex()
        total_solutions = num_placements/factorial(dim_arch[0]*dim_arch[1] - len(cp_vertices))
        
        for i,cgra in enumerate(cgras):
            permutations = unique_permutations(vertices)
            valid_solutions = 0
            for permutation in permutations:
                pe_to_node = {}
                node_to_pe = {}
                for pe,node in enumerate(permutation):
                    if node != -1:
                        node_to_pe[node] = pe
                    pe_to_node[pe] = node
                pes_to_routing = {}
                free_interconnections = copy.deepcopy(cgra.free_connections) 
                for node in order:
                    cost,pes_to_routing,free_interconnections = route(node, pes_to_routing, free_interconnections, in_vertices, node_to_pe, cgra.out_vertices)

                    if cost == 0:
                        break
                if cost == 0:
                    continue

                mapping_is_valid,_ = UtilMapping.mapping_is_valid_2(cp_vertices,order[0],node_to_pe,in_vertices,out_vertices,pes_to_routing)
                if mapping_is_valid:
                    valid_solutions += 1

            print(f"Arch: {archs[i]}. Number of valid solutions: {valid_solutions}. Number of solutions: {total_solutions}. Valid solutions Rate: {valid_solutions/total_solutions}")
            df.loc[len(df)] = [archs[i],file.split('/')[1],valid_solutions/total_solutions]
        print()
    plt.figure(figsize=(7, 4)) 

    g = sns.catplot(
    data=df,
    x='dfg_name', 
    y='valid_sol_rate', 
    hue='Interconnection Style', 
    kind='bar',
    height=4,
    aspect=2
)

    g.fig.suptitle('Valid Solution Rates for Different DFGs and Interconnection Styles', fontsize=18, weight='bold')
    g.set_axis_labels('DFG Name', 'Valid Solutions Rate', fontsize=14, weight='bold')

    # g.set_xticklabels(rotation=45)

    g.fig.subplots_adjust(top=0.9)

    plt.savefig('results/challenges.png')
    plt.show()

if __name__ == "__main__":
    generate_num_placement_possibilities()
    print()
    generate_valid_mapping_rates_3x3()