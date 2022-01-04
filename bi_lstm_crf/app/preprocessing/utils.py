import json
from torch.nn.functional import pad
from torch import tensor

START_TAG = "<START>"
STOP_TAG = "<STOP>"

PAD = "<PAD>"
OOV = "<OOV>"


def save_json_file(obj, file_path):
    with open(file_path, "w", encoding="utf8") as f:
        f.write(json.dumps(obj, ensure_ascii=False))


def load_json_file(file_path):
    with open(file_path, encoding="utf8") as f:
        return json.load(f)

def wordlist_to_idx(ll, word2idx, oov_idx):
    vec = [word2idx.get(c, oov_idx) for c in ll]
    return tensor(vec)

def padder_func(tensor_i, value, max_len = 60):
    '''
    Padder function takes in a one dimensional tensor and post-pads it with a designated value.
    
    Parameters:
        tensor_i -- One dimensional tensor.
        value -- The value to be used as the pad.
        max_len -- The final length of the tensors (default value = 60).
    
    Note that if the length of the tensor exceeds max_len, it will be post-truncated so that the final length is max_len.
    
    Returns:
        List. Padded list of length equal to max_len.
    '''
    return pad(tensor_i, pad = (0, max_len - len(tensor_i)), mode = "constant", value = value).tolist()