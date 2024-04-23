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
import os
import sys
import time

content_head = """
import os
os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
os.environ['FLAGS_group_schedule_tiling_first'] = '1'
os.environ['FLAGS_prim_all'] = 'true'
os.environ['FLAGS_prim_enable_dynamic'] = '1'
os.environ['FLAGS_print_ir'] = '1'
os.environ['FLAGS_enable_pir_api'] = '1'
os.environ['FLAGS_cinn_bucket_compile'] = '1'

import unittest
import numpy as np
import paddle


class CinnMonkeyNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

"""

content_body = """
class TestCinnMonkey(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
"""

content_tail = """
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
        net = CinnMonkeyNet()
        net.eval()
        net = self.apply_to_static(net, use_cinn)
"""

content_main = """
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

def generate_content_neck(tensors_list):
    num_tensors = len(tensors_list)
    forward_function = "    def forward(self"
    for i in range(num_tensors):
        forward_function += f", x{i+1}"
    forward_function += "):\n"

    for i, tensor_name in enumerate(tensors_list):
        forward_function += f"        {tensor_name} = x{i+1}\n"
    forward_function += "\n"

    return forward_function

def generate_content_prepare(shape_list):
    content = ''
    for i, shape in enumerate(shape_list):
        content += f'        self.x{i+1} = paddle.uniform([{shape}], dtype="float32", min=-0.5, max=0.5)\n'
        content += f'        self.x{i+1}.stop_gradient = True\n'
    return content

def generate_content_train(num):
    content = '        out = net('
    for i in range(num):
        content += f'self.x{i+1}, '
    content = content[:-2] + ')'
    return content

def get_content_and_write(dir):
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
    lines = script.file_content.strip().split('\n')

    input_names = []
    input_shapes = []
    net_lines = []
    is_input_line = True
    for line in lines:
        if is_input_line:
            if line.startswith('#'):
                is_input_line = False
                net_lines.append(line)
            if line.startswith('tensor'):
                input_names.append(line.split()[0])
                start_index = line.find("((")
                end_index = line.find("))")
                if start_index != -1 and end_index != -1:
                    input_shapes.append(line[start_index + 2:end_index])
        else:
            net_lines.append(line)

    out_tensor_name = None
    for line in reversed(net_lines):
        if not line.startswith('assert'):
            out_tensor_name = line.split()[0]
            break

    # dir = "/home/aistudio/data"
    # dir = "/dev/shm/test"
    current_datetime = datetime.datetime.now()
    curr_time = current_datetime.strftime("%Y-%m-%d_%H-%M-%S-%f")
    file_name = dir + "/testcase_" + curr_time + ".py"
    with open(file_name, "w") as f:
        f.write(content_head)
        f.write(generate_content_neck(input_names))
        for line in net_lines:
            if line.startswith("assert"):
                parts = line.split(" ")
                parts[1] = "tuple(" + parts[1] + ")"
                line = " ".join(parts)
            f.write("        " + line + "\n")
        f.write("        return " + out_tensor_name + "\n\n")
        f.write(content_body)
        f.write(generate_content_prepare(input_shapes))
        f.write(content_tail)
        f.write(generate_content_train(len(input_names)))
        f.write(content_main)

def check_file_existence(directory, filename):
    """
    Check if the specified file exists in the given directory.

    Args:
    - directory (str): The directory to search for the file.
    - filename (str): The name of the file to check for.

    Returns:
    - bool: True if the file exists, False otherwise.
    """
    file_path = os.path.join(directory, filename)
    return os.path.isfile(file_path)

def count_python_files(directory):
    """
    Count the number of Python files in the specified directory.

    Args:
    - directory (str): The directory to search for Python files.

    Returns:
    - int: The number of Python files in the directory.
    """
    python_files = [file for file in os.listdir(directory) if file.endswith(".py")]
    return len(python_files)

def generate_test_cases_until_file_exists(directory, flag_file, max_python_files):
    """
    Continuously generate test cases until the specified file exists.
    
    Args:
    - directory (str): The directory to search for Python files.
    - flag_file (str): The name of the file to check for existence.
    - max_python_files (int): The maximum number of Python files allowed before pausing.

    Returns:
    - None
    """
    while not check_file_existence(directory, flag_file):
        python_file_count = count_python_files(directory)

        if python_file_count < max_python_files:
            print("py", python_file_count)
            get_content_and_write(directory)
            # time.sleep(0.05)
        else:
            time.sleep(0.5)

def process_arguments():
    if len(sys.argv) < 4:
        arg_dir = "/dev/shm/test"
        arg_flag = "stop"
        arg_max = 10
    else:
        arg_dir = sys.argv[1]
        arg_flag = sys.argv[2]
        arg_max = int(sys.argv[3])

    return arg_dir, arg_flag, arg_max

def main():
    directory, flag_file, max_python_files = process_arguments()
    print("dir:", directory)
    print("flag:", flag_file)
    print("max:", max_python_files)

    if not os.path.exists(directory):
        os.makedirs(directory)
    generate_test_cases_until_file_exists(directory, flag_file, max_python_files)

if __name__ == "__main__":
    main()
