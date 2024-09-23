import unittest
from src.enums.enum_operation import EnumOperation
class TestEnumOperation(unittest.TestCase):
    def test_get_enum_operation_by_id(self):
        self.assertEqual(EnumOperation.enum_operation_by_id(0), EnumOperation.ADD)
        self.assertEqual(EnumOperation.enum_operation_by_id(1), EnumOperation.MULT)
        self.assertEqual(EnumOperation.enum_operation_by_id(2), EnumOperation.CONST)
        self.assertEqual(EnumOperation.enum_operation_by_id(3), EnumOperation.LOAD)
        self.assertEqual(EnumOperation.enum_operation_by_id(4), EnumOperation.OUTPUT)

    def test_get_enum_operation_by_type(self):
        self.assertEqual(EnumOperation.enum_operation_by_type('add'), EnumOperation.ADD)
        self.assertEqual(EnumOperation.enum_operation_by_type('mul'), EnumOperation.MULT)
        self.assertEqual(EnumOperation.enum_operation_by_type('load'), EnumOperation.LOAD)
        self.assertEqual(EnumOperation.enum_operation_by_type('const'), EnumOperation.CONST)
        self.assertEqual(EnumOperation.enum_operation_by_type('output'), EnumOperation.OUTPUT)

if __name__ == "__main__":
    unittest.main()