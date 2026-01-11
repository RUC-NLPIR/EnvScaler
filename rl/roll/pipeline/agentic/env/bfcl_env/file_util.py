import json

def _read_json(read_file_path):
    with open(read_file_path,encoding='utf-8') as file:
        raw_data=json.load(file)
    return raw_data


def _read_jsonl(read_file_path):
    data = []
    with open(read_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def _read_txt(read_file_path):
    with open(read_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content
    

def load_file(read_file_path):
    if read_file_path.endswith('.json'):
        return _read_json(read_file_path)
    elif read_file_path.endswith('.jsonl'):
        return _read_jsonl(read_file_path)
    elif read_file_path.endswith('.txt'):
        return _read_txt(read_file_path)
    else:
        raise ValueError('file_type is not supported')