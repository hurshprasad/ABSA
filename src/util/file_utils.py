def write_to_file(file_name, content):
    file = open(file_name, 'w')
    file.write(content)
    file.close()


def read(file_name):
    train_file = open(file_name)
    content = train_file.read()
    return content


def check_saved_file(file):
    import os
    return os.path.exists(file)
