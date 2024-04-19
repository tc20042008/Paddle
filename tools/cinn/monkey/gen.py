from dataclasses import dataclass
from collections import namedtuple
from typing import Generator, Union
from defensive_list import DList
from dag_generator import PickWeight
from script import Script
import dag_generator
import dim_eq1_generator as dim_eq1_generator
import dims_eq1_generator as dims_eq1_generator
import op_name_generator as op_name_generator
import tensor_name_generator as tensor_name_generator
from tensor_name_generator import TensorNameGenRequirement
import shape_signature_inferer as shape_signature_inferer
from shape_signature_inferer import StaticDim
import instruction_util as instruction_util
import op_call_code_gen
from paddle_eager_generator import PaddleEagerGenerator
from numpy_generator import NumpyGenerator
from unit_test_case_spec import (
    UnitTestCaseRequirement,
    UnitTestCaseSpec,
    GenerateRandomUnitTestCaseSpec,
    GetAblatedUnitTestCaseSpec
)
import datetime

content_head = """
import unittest
import numpy as np
import paddle


class CINNCosSubGraphNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        tensor1 = x
"""

content_body = """
class TestCinnCos(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
"""

content_tail = """
        self.x.stop_gradient = True

    def apply_to_static(self, net, use_cinn, input_spec=None):
        build_strategy = paddle.static.BuildStrategy()
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = CINNCosSubGraphNet()
        net.eval()
        net = self.apply_to_static(net, use_cinn)
        out = net(self.x)
        return out

    def test_train(self):
        cinn_out = self.train(use_cinn=True)
        dy_out = self.train(use_cinn=False)

        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-6)


if __name__ == '__main__':
    unittest.main()
"""

def GenerateUnitTestCaseSpec(
    unit_test_case_requirement: UnitTestCaseRequirement
) -> UnitTestCaseSpec:
    return GenerateRandomUnitTestCaseSpec(
        requirement=unit_test_case_requirement
    )

def modify_assert_line(text):
    lines = text.split("\n")
    text = ""
    for line in lines:
        if line.startswith("assert"):
            parts = line.split(" ")
            parts[1] = "tuple(" + parts[1] + ")"
            line = " ".join(parts)
        text += line + "\n"
    return text

if __name__ == '__main__':
    unit_test_case_requirement=UnitTestCaseRequirement(
        dag_gen_requirement=dag_generator.DAGGenRequirement(
            min_num_sources=0,
            max_num_sources=0,
            pick_probability=dag_generator.DAGGenTypePickProbability()
        ),
        dims_eq1_gen_requirement=dims_eq1_generator.DimsEq1GenRequirement(
            dims_eq1_probability=[0.1, 0.15, 0.2]
        ),
        op_name_gen_requirement=op_name_generator.OpNameGenRequirement(),
        tensor_name_gen_requirement=TensorNameGenRequirement(),
        dim_size_requirement=shape_signature_inferer.DimSizeRequirement(
            dim_size=[StaticDim(128), StaticDim(64), StaticDim(32)]
        )
    )
    unit_test_case_spec = GenerateUnitTestCaseSpec(
        unit_test_case_requirement=unit_test_case_requirement
    )

    # numpy_gen = NumpyGenerator()

    paddle_eager_gen = PaddleEagerGenerator()
    # script = numpy_gen.Generate(unit_test_case_spec.patched_instruction_code_gen_spec)
    # print("import numpy")
    # print(script.file_content)

    script = paddle_eager_gen.Generate(unit_test_case_spec.patched_instruction_code_gen_spec)
    # print("import paddle")
    # print(script.file_content)
    net_content = modify_assert_line(script.file_content)

    outpu_dir = "/home/aistudio/data"
    current_datetime = datetime.datetime.now()
    curr_time = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    with open(outpu_dir + "/" + curr_time + "_content.py", "w") as f:
        f.write(net_content)

    lines = net_content.strip().split('\n')
    input_shape = None
    for line in lines:
        if line.startswith('tensor1'):
            start_index = line.find("((")
            end_index = line.find("))")
            if start_index != -1 and end_index != -1:
                input_shape = line[start_index + 2:end_index]
            break

    out_tensor_name = None
    for line in reversed(lines):
        if not line.startswith('assert'):
            out_tensor_name = line.split()[0]
            break

    # current_datetime = datetime.datetime.now()
    file_name = outpu_dir + "/" + curr_time + "_testcase.py"
    with open(file_name, "w") as f:
        f.write(content_head)
        for line in lines:
            if not line.startswith('tensor1 '):
                f.write("        " + line + "\n")
        f.write("        return " + out_tensor_name + "\n\n")
        f.write(content_body)
        f.write('        self.x = paddle.uniform([' + input_shape + '], dtype="float32", min=-0.5, max=0.5)')
        f.write(content_tail)


#    ablated_unit_test_case_spec = GetAblatedUnitTestCaseSpec(
#        instructions=unit_test_case_spec.instructions,
#        requirement=unit_test_case_requirement,
#        bottom_up_ablation_size=-1,
#        component_ablation_size=-1,
#    )
#    print("#", "*"*80)
#    print("# full ablated")
#    print("#", "*"*80)
#    script = numpy_gen.Generate(ablated_unit_test_case_spec.patched_instruction_code_gen_spec)
#    print("import paddle")
#    print(script.file_content)
#
#
#    ablated_unit_test_case_spec = GetAblatedUnitTestCaseSpec(
#        instructions=unit_test_case_spec.instructions,
#        requirement=unit_test_case_requirement,
#        bottom_up_ablation_size=len(unit_test_case_spec.instructions)/2,
#        component_ablation_size=-1,
#    )
#    print("#", "*"*80)
#    print("# half ablated")
#    print("#", "*"*80)
#    script = numpy_gen.Generate(ablated_unit_test_case_spec.patched_instruction_code_gen_spec)
#    print("import paddle")
#    print(script.file_content)
