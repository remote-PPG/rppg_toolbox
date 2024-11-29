import hashlib as __hashlib

def generate_file_hash(file_path):
    with open(file_path, 'rb') as file:
        file_contents = file.read()
        file.close()
    return __hashlib.sha256(file_contents).hexdigest()
def generate_dict_hash(d:dict):
    dict_str = str(d)
    return __hashlib.sha256(dict_str.encode()).hexdigest()