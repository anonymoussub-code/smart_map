



from pathlib import Path
import pygraphviz as pgv
import networkx as nx
path = Path("benchmarks/changed_MCTS_benchmark/")

files = [str(file) for file in path.rglob('*.dot') if file.is_file()]
print(len(files))
count = 0
for file in files:
    G1 = nx.drawing.nx_pydot.read_dot(file)
    if len(G1.nodes()) <= 64:
        count+= 1
        print(file,len(G1.nodes()),len(G1.edges()))
print(count)