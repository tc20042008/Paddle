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
import os
import sys
import time

def GenerateUnitTestCaseSpec(
    unit_test_case_requirement: UnitTestCaseRequirement
) -> UnitTestCaseSpec:
    return GenerateRandomUnitTestCaseSpec(
        requirement=unit_test_case_requirement
    )

def get_content():
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
    # script = numpy_gen.Generate(unit_test_case_spec.patched_instruction_code_gen_spec)

    paddle_eager_gen = PaddleEagerGenerator()
    script = paddle_eager_gen.Generate(unit_test_case_spec.patched_instruction_code_gen_spec)
    # print("import numpy")
    # print(script.file_content)
    return script.file_content

counter = 0

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
            gen(directory)  # Call your existing gen() function to generate test cases
            time.sleep(0.05)
        else:
            time.sleep(0.05)

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

def gen(path):
    # Placeholder for the gen() function. Implement your test case generation logic here.
    # print("Generating test case...")
    global counter
    while True:
        file_name = os.path.join(path, f"test_case_b_{counter}.py")
        counter += 1
        if not os.path.isfile(file_name):
            break
    print("gen", counter-1)
    with open(file_name, "w") as f:
        f.write("import paddle")
        f.write(modify_assert_line(get_content()))
        # f.write("import numpy")
        # f.write(get_content())

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
