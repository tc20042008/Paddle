import os
import shutil
import time

def process_py_files_until_file_exists(directory, flag_file):
    """
    Continuously process Python files in the specified directory until the flag file exists.

    Args:
    - directory (str): The directory to search for Python files.
    - flag_file (str): The name of the flag file to check for existence.

    Returns:
    - int: The number of times the test function was called.
    """
    selection_dir = os.path.join(directory, "selection")
    if not os.path.exists(selection_dir):
        os.makedirs(selection_dir)
    cnt = 0
    cnt_slc = 0
    while not check_file_existence(directory, flag_file):
        py_files = [file for file in os.listdir(directory) if file.endswith(".py")]

        if py_files:
            for py_file in py_files:
                file_path = os.path.join(directory, py_file)
                with open(file_path) as file:
                    test_case = file.read()
                    try:
                        # print(test_case)
                        exec(test_case)
                        os.remove(file_path)
                        # time.sleep(5)
                    except AssertionError:
                        shutil.move(file_path, selection_dir)
                        cnt_slc += 1
                cnt += 1
                print(f"\rTested: {cnt}. Selection: {cnt_slc}.", end="")
                # os.remove(file_path)
        else:
            time.sleep(0.05)  # Sleep for 50 milliseconds (adjust as needed)

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

if __name__ == "__main__":
    call_count = process_py_files_until_file_exists("/dev/shm/test", "stop")
