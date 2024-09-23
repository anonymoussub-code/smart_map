import unittest
from util_interconnections import UtilInterconnections

class TestUtilInterconnections(unittest.TestCase):
    '''
    General variables:
    dim_cgra = A tuple that represents the dimension of a CGRA (5x5), where n is the number of rows and m is the number of columns.
    '-> [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
         (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
         (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
         (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
         (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]

    pe_pos = A tuple, wich represents a pe position
    '''
    def test_generate_interconnection_by_adj_positions_and_pe_pos(self):
        '''
        variables:
        adj_positions = A tuple list, [(-1, 0), (1, 0), (0, 1), (0, -1)], relative positions 
        target_interconnection_list = A tuple list, which represents the expected output
        actual_interconenction_list = A tuple list, which represents the actual output
        '->[(1-1,1+0),(1+1,1+0),(1+0,1+1),(1+0,1-1)]

        test:
        Compare expected output with actual output
        '''
        dim_cgra = (5,5)        
        pe_pos = (1,1)
        adj_positions = [(-1, 0), (1, 0), (0, 1), (0, -1)] 
        target_interconnection_list = [(0,1),(2,1),(1,2),(1,0)]
        actual_interconenction_list = UtilInterconnections.generate_interconnection_by_adj_positions_and_pe_pos(adj_positions,pe_pos,dim_cgra)
        self.assertEqual(target_interconnection_list,actual_interconenction_list)

    def test_generate_interconnection_for_pe_pos_is_out_of_border(self):
        '''
        variables:
        pe_pos = Is out of CGRA border
        adj_positions = A tuple list, [(-1, 0), (1, 0), (0, 1), (0, -1)], relative positions 
        target_interconnection_list = A tuple list, which represents the expected output
        actual_interconenction_list = A tuple list, which represents the actual output

        test:
        Compare expected output with actual output, To check if the function is adding invalid positions to the list
        '''
        
        dim_cgra = (5,5)        
        pe_pos = (6,6)
        adj_positions = [(-1, 0), (1, 0), (0, 1), (0, -1)] 
        target_interconnection_list = []
        actual_interconenction_list = UtilInterconnections.generate_interconnection_by_adj_positions_and_pe_pos(adj_positions,pe_pos,dim_cgra)
        self.assertEqual(target_interconnection_list,actual_interconenction_list)

    def test_generate_mesh_interconnection(self):
        '''
        adj_positions:
        [(-1,0),(1,0),(0,1),(0,-1)]

        variables:
        target_interconnection_list = A tuple list, which represents the expected output
        actual_interconenction_list = A tuple list, which represents the actual output
        '->Same as the previous function

        test:
        Compare expected output with actual output
        '''
        dim_cgra = (5,5)
        pe_pos = (1,1) 
        target_interconnections_list = target_interconnection_list = [(0,1),(2,1),(1,2),(1,0)]
        actual_interconnections_list = UtilInterconnections.generate_mesh_interconnection_by_pe_pos(pe_pos,dim_cgra)
        self.assertEqual(target_interconnections_list,actual_interconnections_list)

    def test_generate_diagonal_interconnection(self):
        '''
        adj_positions:
        [(-1,-1),(-1,1),(1,-1),(1,1)]

        variables:
        target_interconnection_list = A tuple list, which represents the expected output
        actual_interconenction_list = A tuple list, which represents the actual output
        '->[(1-1,1-1),(1-1,1+1),(1+1,1-1),(1+1,1+1)]

        test:
        Compare expected output with actual output
        '''
        dim_cgra = (5,5)
        pe_pos = (1,1) 
        target_interconnections_list = [(0,0),(0,2),(2,0),(2,2)]
        actual_interconnections_list = UtilInterconnections.generate_diagonal_interconnection_by_pe_pos(pe_pos,dim_cgra)
        self.assertEqual(target_interconnections_list,actual_interconnections_list)


    def test_generate_one_hop_interconnection(self):
        '''
        adj_positions:
        [(-2,0),(2,0),(0,2),(0,-2)]

        variables:
        target_interconnection_list = A tuple list, which represents the expected output
        actual_interconenction_list = A tuple list, which represents the actual output
        '->[(2-2,2+0),(2+2,2+0),(2+0,2+2),(2+0,2-2)]
        
        test:
        Compare expected output with actual output
        '''
        dim_cgra = (5,5)
        pe_pos = (2,2) 
        target_interconnections_list = [(0,2),(4,2),(2,4),(2,0)]
        actual_interconnections_list = UtilInterconnections.generate_one_hop_interconnection_by_pe_pos(pe_pos,dim_cgra)

    def test_generate__toroidal_interconnection_pe_pos_is_border(self):
        '''
        adj_positions : 
        [(0,dim_cgra[1]-1),(0.-(dim_cgra[1]-1)),(dim_cgra[0]-1,0),(-(dim_cgra[0]-1),0)]
        '->In this specific case, where the pe_pos is (4,4)
            '->[(0,4),(0,-4),(4,0),(-4,0)]
        
        variables:
        target_interconnection_list = A tuple list, which represents the expected output
        actual_interconenction_list = A tuple list, which represents the actual output
        '->[(4+0,4+3),(4+0,4-4),(4+4,4+0),(4-4,4+0)]

        test:   
        Compare expected output with actual output
        '''

        dim_cgra = (5,5)
        pe_pos = (4,4) 
        target_interconnections_list = [(4,0),(0,4)] #Without invalid positions
        actual_interconnections_list = UtilInterconnections.generate_toroidal_interconnection_by_pe_pos(pe_pos,dim_cgra)
        self.assertEqual(target_interconnections_list,actual_interconnections_list)


    def test_generate__toroidal_interconnection_pe_pos_is_not_border(self):
        '''
        variables:
        pe_pos = Is not on the border
        target_interconnection_list = A tuple list, which represents the expected output
        actual_interconenction_list = A tuple list, which represents the actual output
        
        test:   
        Compare expected output with actual output
        '''
        dim_cgra = (5,5)
        pe_pos = (2,2) 
        target_interconnections_list = []
        actual_interconnections_list = UtilInterconnections.generate_toroidal_interconnection_by_pe_pos(pe_pos,dim_cgra)

    def test_generate_crossbar_interconnection(self):
        '''
        variables:
        target_interconnection_list = A tuple list, which represents the expected output
        actual_interconenction_list = A tuple list, which represents the actual output
        '->All pe_pos in a CGRA 5x5, except the (2,2), wich is the pe_pos referencial
        
        test:
        Compare expected output with actual output
        '''
        dim_cgra = (5,5)
        pe_pos = (2,2) 
        target_interconnections_list = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), (2, 1), (2, 3), (2, 4), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]
        actual_interconnections_list = UtilInterconnections.generate_crossbar_interconnection_by_pe_pos(pe_pos,dim_cgra)
    


if __name__ == "__main__":
    unittest.main()