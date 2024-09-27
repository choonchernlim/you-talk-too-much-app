def read_file(file_path) -> str:
    with open(file_path, 'r') as file:
        text = file.read()

    return text


def write_file(file_path, text):
    with open(file_path, 'w') as file:
        file.write(text)


def append_file(file_path, text):
    with open(file_path, 'a') as file:
        file.write(text)
