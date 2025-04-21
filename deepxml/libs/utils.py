from typing import Union
from scipy.sparse import spmatrix
from numpy import ndarray

import os
import math
import functools
import numpy as np
from tqdm import tqdm
from contextlib import contextmanager
from scipy.sparse import save_npz
import torch.multiprocessing as mp
from transformers import AutoTokenizer
from xclib.utils.sparse import ll_to_sparse
from xclib.data.data_utils import read_corpus, write_sparse_file


def compute_depth_of_tree(n: int, s: int) -> int:
    """Get depth of tree 

    Args:
        n (int): Total number of items at root node 
        s (int): Cluster size at the leaf node 

    Returns:
        int: Depth of tree
    """
    return int(math.ceil(math.log(n / s) / math.log(2)))


def get_filter_map(fname: str) -> Union[ndarray, None]:
    if fname is not None:
        return np.loadtxt(fname).astype(np.int)
    else:
        return None


def filter_predictions(pred: spmatrix, mapping: ndarray=None) -> spmatrix:
    if mapping is not None and len(mapping) > 0:
        pred[mapping[:, 0], mapping[:, 1]] = 0
        pred.eliminate_zeros()
    return pred


def save_predictions(pred: spmatrix, fname: str) -> None:
    save_npz(fname, pred.tocsr())


def epochs_to_iterations(n: int, n_epochs: int, bsz: int) -> int:
    """A helper function to convert between epoch and iterations or steps
    * Useful for optimizer
    
    Args:
        n (int): number of data points
        n_epochs (int): number of epochs
        bsz (int): batch size

    Returns:
        int: number of iterations or steps
    """
    return n_epochs * math.ceil(n//bsz)


@contextmanager
def evaluating(net):
    """
    A context manager to temporarily set the model to evaluation mode.
    
    It saves the current training state of the model, switches to eval mode,
    and then restores the original state after the block is executed.
    """
    org_mode = net.training  # Save the current mode (True if training, False if eval)
    net.eval()  # Set to eval mode
    try:
        yield net
    finally:
        # Restore the model's original mode
        if org_mode:
            net.train()


def _tokenize_one(batch_input):
    tokenizer, batch_corpus = batch_input
    temp = tokenizer(batch_corpus)
    return (temp['input_ids'], temp['attention_mask'])


def _tokenize_mp(corpus, tokenizer, num_threads, bsz=10000): 
    batches = [(tokenizer, corpus[i: i + bsz]) for i in range(0, len(corpus), bsz)]

    pool = mp.Pool(num_threads)
    batch_tokenized = pool.map(_tokenize_one, batches)
    pool.close()

    input_ids = np.vstack([x[0] for x in batch_tokenized])
    attention_mask = np.vstack([x[1] for x in batch_tokenized])

    del batch_tokenized 

    return input_ids, attention_mask


def tokenize_corpus(
        corpus: str, 
        tokenizer_type: str,
        tokenization_dir: str,
        max_len: int, 
        prefix: str,
        do_lower_case: bool=True,
        num_threads: int=6, 
        batch_size: int=100000):
    """Tokenize text in a given file and dump it on disk

    Args:
        corpus (str): Path of the corpus (each line is treated a separate chunk)
        tokenizer_type (str): Tokenizer type (e.g., bert-base-uncased)
        tokenization_dir (str): Dump tokenized files in this directory
        max_len (int): max tokenization length
        prefix (str): use it for output file name
        do_lower_case (bool, optional): lowercase text? Defaults to True.
        num_threads (int, optional): Threads for multi-processing. Defaults to 6.
        batch_size (int, optional): Defaults to 100000
            Consider these many documents at a time.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_type,
        do_lower_case=do_lower_case)

    tokenizer = functools.partial(
        tokenizer.batch_encode_plus,
        add_special_tokens=True,              # Add '[CLS]' and '[SEP]'
        max_length=max_len,                   # Pad & truncate all sentences.
        padding='max_length',
        return_attention_mask=True,           # Construct attn. masks.
        return_tensors='np',                  # Return numpy tensors.
        truncation=True
    )

    # TODO: Avoid loading full text at once
    with open(corpus, "r", encoding='latin') as fp:
        corpus = [x.strip() for x in fp.readlines()]

    ind = np.lib.format.open_memmap(
        os.path.join(tokenization_dir, f"{prefix}_input_ids.npy"),
        shape=(len(corpus), max_len), 
        mode='w+',
        dtype='int64')

    mask = np.lib.format.open_memmap(
        f"{tokenization_dir}/{prefix}_attention_mask.npy",
        shape=(len(corpus), max_len), 
        mode='w+',
        dtype='int64')

    ind[:], mask[:] = 0, 0

    for i in tqdm(range(0, len(corpus), batch_size), desc="Tokenizing.."):
        _ids, _mask = _tokenize_mp(
            corpus[i: i + batch_size], tokenizer, num_threads)
        ind[i: i + _ids.shape[0], :] = _ids
        mask[i: i + _ids.shape[0], :] = _mask
    ind.flush()
    mask.flush()


def extract_text_labels(
        in_fname: str, 
        op_tfname: str, 
        op_lfname: str=None, 
        fields: list[str]=["title"], 
        num_labels: int=-1):
    """Extract text and labels from json.gz file

    Args:
        in_fname (str): Input file
        op_tfname (str): Dump text in this file
        op_lfname (str, optional): Dump labels in this file. Defaults to None.
            None is useful if label file does not apply
        fields (list[str], optional): Defaults to ["title"].
            concatenate these fields (e.g., ["title"] or ["title", "description"])
        num_labels (int, optional): #labels in the data. Defaults to -1.
            Can be useful when last few labels are not available in some file
    """
    labels = []
    with open(op_tfname, 'w', encoding='latin') as fp:
        for line in read_corpus(in_fname):
            t = ""
            labels.append(line['target_ind'])
            for f in fields:
                t += f"{line[f]} "
            fp.write(t.strip() + "\n")
    if num_labels == -1:
        max_ind = max([max(item) for item in labels])
        print("num_labels is -1; will be determining index from json.gz")
        num_labels = max_ind
    if op_lfname is not None:
        labels = ll_to_sparse(
            labels, shape=(len(labels), num_labels))
        write_sparse_file(labels, op_lfname, header=True)


def count_num_labels(in_fname: str) -> int:
    """Count number of rows in lbl.json.gz file

    Args:
        in_fname (str): The raw data file

    Returns:
        int: Number of rows in json.gz file
    """
    return sum([1 for _ in read_corpus(in_fname)])
