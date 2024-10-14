import os
import pytest
from phasefieldx import (
    get_filenames_in_folder,
    get_foldernames_in_folder,
    find_files_with_extension,
    read_last_line,
    read_last_lines,
    replace_text,
    delete_specific_files,
    delete_folders_with_contents,
    prepare_simulation,
    delete_folder,
    append_results_to_file
)


@pytest.fixture
def setup_directory(tmp_path):
    # Create temporary directory structure
    folder = tmp_path / "test_folder"
    folder.mkdir()

    # Create some files for testing
    (folder / "file1.txt").write_text("This is file 1.")
    (folder / "file2.txt").write_text("This is file 2.")
    (folder / "file3.log").write_text("This is a log file.")
    (folder / "subfolder").mkdir()

    return str(folder)


def test_get_filenames_in_folder(setup_directory):
    filenames = get_filenames_in_folder(setup_directory)
    assert sorted(filenames) == ["file1.txt", "file2.txt", "file3.log"]


def test_get_foldernames_in_folder(setup_directory):
    foldernames = get_foldernames_in_folder(setup_directory)
    assert foldernames == ["subfolder"]


def test_find_files_with_extension(setup_directory):
    matching_files = find_files_with_extension(setup_directory, ".txt")
    expected_files = [os.path.join(setup_directory, "file1.txt"), os.path.join(setup_directory, "file2.txt")]
    assert sorted(matching_files) == sorted(expected_files)


def test_read_last_line(setup_directory):
    last_line = read_last_line(os.path.join(setup_directory, "file1.txt"))
    assert last_line == "This is file 1."


def test_read_last_lines(setup_directory):
    lines = read_last_lines(os.path.join(setup_directory, "file1.txt"), 1)
    assert lines == ["This is file 1."]


def test_replace_text(setup_directory):
    original_file = os.path.join(setup_directory, "file1.txt")
    target_file = os.path.join(setup_directory, "file1_replaced.txt")
    replacements = [("file 1", "replaced file 1")]

    success = replace_text(original_file, target_file, replacements)
    assert success

    with open(target_file, 'r') as f:
        content = f.read()
        assert "replaced file 1." in content


def test_delete_specific_files(setup_directory):
    delete_specific_files(setup_directory, ['.log'])
    assert not os.path.exists(os.path.join(setup_directory, "file3.log"))


def test_delete_folders_with_contents(setup_directory):
    delete_folders_with_contents(setup_directory, ["subfolder"])
    assert not os.path.exists(os.path.join(setup_directory, "subfolder"))


def test_prepare_simulation(setup_directory):
    prepare_simulation(setup_directory, "results")
    assert os.path.exists(os.path.join(setup_directory, "results"))


def test_delete_folder(setup_directory):
    test_folder = os.path.join(setup_directory, "temp_folder")
    os.mkdir(test_folder)
    temp_file_path = os.path.join(test_folder, "temp_file.txt")
    with open(temp_file_path, 'w') as temp_file:
        temp_file.write("Temporary file.")

    delete_folder(test_folder)
    assert not os.path.exists(test_folder)


def test_append_results_to_file(tmp_path):
    output_file_path = tmp_path / "results.txt"  # Create a temporary file path
    header = "Step\tValue1\tValue2"

    # Clear the output file if it already exists
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    # First call: appending header and first line
    append_results_to_file(str(output_file_path), header, 1, 0.5, 1.2)

    # Verify the file content after the first append
    with open(output_file_path, 'r') as file:
        lines = file.readlines()
        assert len(lines) == 2  # Expecting 2 lines (the header + first data line)
        assert lines[0].strip() == "Step\tValue1\tValue2"  # Check header
        assert lines[1].strip() == "1\t0.5\t1.2"  # Check the appended data line

    # Second call: appending second line
    append_results_to_file(str(output_file_path), header, 2, 0.6, 1.3)

    # Verify the file content after the second append
    with open(output_file_path, 'r') as file:
        lines = file.readlines()
        assert len(lines) == 3  # Expecting 3 lines (header + first + second data line)
        assert lines[2].strip() == "2\t0.6\t1.3"  # Check the appended data line
