import unittest

from src.utils.util_graph import UtilGraph

class TestUtilGraph(unittest.TestCase):

    def test_creation_dict_in_vertices(self):
        vertices = [0,1,2,3]
        edges = [(0,1),(1,2),(0,3),(3,2)]
        in_vertices = UtilGraph.generate_dict_in_or_out_vertices(vertices,edges,'in')
        target_dict = {0:[],1:[0],2:[3,1],3:[0]}
        assert in_vertices.keys() == target_dict.keys()
        for key in in_vertices.keys():
            assert(sorted(in_vertices[key]) == sorted((target_dict[key])))  
    
    def test_creation_dict_out_vertices(self):
        vertices = [0,1,2,3]
        edges = [(0,1),(1,2),(0,3),(3,2)]
        in_vertices = UtilGraph.generate_dict_in_or_out_vertices(vertices,edges,'out')
        target_dict = {0:[1,3],1:[2],2:[],3:[2]}
        assert in_vertices.keys() == target_dict.keys()
        for key in in_vertices.keys():
            assert(sorted(in_vertices[key]) == sorted((target_dict[key])))
    
    def test_creation_edges_index_by_edges(self):
        edges = [(0,1),(1,2),(0,3),(3,2)]
        target_edges_index = [[0,1,0,3],[1,2,3,2]]
        edges_index = UtilGraph.generate_edges_index_by_edges(edges)
        self.assertListEqual(edges_index,target_edges_index)

    def test_the_function_init_dict_node_to_something_with_number_vertices(self):
        '''
        variables: num_vertices = random number of vertices , target_node_to_pe = Dictionary created manually to simulate the target output of the function
        real_node_to_pe = Actual function output

        test: Compare the function output (real_node_to_pe) with the variable "target_node_to_pe"
        '''
        num_vertices = 5
        target_node_to_pe = {0 : -1, 1:-1, 2:-1, 3:-1, 4:-1}
        real_node_to_pe = UtilGraph.init_dict_node_to_something(num_vertices)
        self.assertEqual(target_node_to_pe,real_node_to_pe)
       
    def test_the_function_init_dict_node_to_something_with_list_vertices(self): 
        '''
        variables: list_vertices = A random vertices list, target_node_to_pe = Dictionary created manually to simulate the target output of the function
        real_node_to_pe = Actual function output

        test: Compare the function output (real_node_to_pe) with the variable "target_node_to_pe"
        '''
        list_vertices = [4,5,6,10,19]
        target_node_to_pe = {4 : -1, 5:-1, 6:-1, 10:-1, 19:-1} 
        real_node_to_pe = UtilGraph.init_dict_node_to_something(list_vertices)
        self.assertEqual(target_node_to_pe,real_node_to_pe)

    def test_the_function_reset_vertices_labels_for_separate_outputs(self):

        '''
        variables: list_vertices = A random vertices list
        target_list_vertices,target_dict_real_to_reset and target_dict_reset_to_real are respectively a list and 2 dictionaries, manually created to simulate the output of the fuction

        test: Tests each of the 3 function outputs separately, comparing with the created variables
        actual_vertices, actual_dict_real_to_reset, actual_dict_reset_to_real = are the actual outputs of the function
        '''
        list_vertices = [10,1,20,89,4,52]
        target_list_vertices = [0,1,2,3,4,5] 
        target_dict_real_to_reset = {10:0, 1:1, 20:2, 89:3, 4:4, 52:5} 
        target_dict_reset_to_real = {0 : 10, 1: 1, 2:20, 3:89, 4:4, 5:52} 
        actual_vertices, actual_dict_real_to_reset, actual_dict_reset_to_real = UtilGraph.reset_vertices_labels(list_vertices) 
        self.assertEqual(actual_vertices,target_list_vertices) 
        self.assertEqual(actual_dict_real_to_reset,target_dict_real_to_reset)
        self.assertEqual(actual_dict_reset_to_real,target_dict_reset_to_real)

    def test_the_function_transform_edges_labels_by_dict(self):
        '''
        variables :old_to_new_nodes = A random dictonary, edges = A list of tuples, in which the tuple values are some of the dictionary values
        target_new_edges = Dictionary created manually to simulate the target output of the function
        actual_new_edges = Actual function output

        Test: Compare the function output (actual_new_edges) with the variable "target_new_edges"
        '''
        edges = [(0,7),(3,10),(10,0),(7,5)]
        old_to_new_nodes = {0:1, 3:2, 10:3, 5:4, 7:5, 4:6} 
        target_new_edges = [(1,5),(2,3),(3,1),(5,4)] 
        actual_new_edges = UtilGraph.transform_edges_labels_by_dict(edges,old_to_new_nodes) 
        self.assertEqual(target_new_edges,actual_new_edges)


if __name__ == "__main__":
    unittest.main()


    
