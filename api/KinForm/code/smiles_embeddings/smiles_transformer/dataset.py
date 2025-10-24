import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .enumerator import SmilesEnumerator

PAD = 0
MAX_LEN = 220
# Split SMILES into words
def split(sm):
    '''
    function: Split SMILES into words. Care for Cl, Br, Si, Se, Na etc.
    input: A SMILES
    output: A string with space between words
    '''
    arr = []
    i = 0
    while i < len(sm)-1:
        if not sm[i] in ['%', 'C', 'B', 'S', 'N', 'R', 'X', 'L', 'A', 'M', \
                        'T', 'Z', 's', 't', 'H', '+', '-', 'K', 'F']:
            arr.append(sm[i])
            i += 1
        elif sm[i]=='%':
            arr.append(sm[i:i+3])
            i += 3
        elif sm[i]=='C' and sm[i+1]=='l':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='C' and sm[i+1]=='a':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='C' and sm[i+1]=='u':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='B' and sm[i+1]=='r':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='B' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='B' and sm[i+1]=='a':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='B' and sm[i+1]=='i':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='S' and sm[i+1]=='i':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='S' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='S' and sm[i+1]=='r':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='N' and sm[i+1]=='a':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='N' and sm[i+1]=='i':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='R' and sm[i+1]=='b':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='R' and sm[i+1]=='a':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='X' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='L' and sm[i+1]=='i':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='A' and sm[i+1]=='l':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='A' and sm[i+1]=='s':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='A' and sm[i+1]=='g':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='A' and sm[i+1]=='u':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='M' and sm[i+1]=='g':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='M' and sm[i+1]=='n':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='T' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='Z' and sm[i+1]=='n':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='s' and sm[i+1]=='i':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='s' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='t' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='H' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='+' and sm[i+1]=='2':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='+' and sm[i+1]=='3':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='+' and sm[i+1]=='4':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='-' and sm[i+1]=='2':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='-' and sm[i+1]=='3':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='-' and sm[i+1]=='4':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='K' and sm[i+1]=='r':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='F' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        else:
            arr.append(sm[i])
            i += 1
    if i == len(sm)-1:
        arr.append(sm[i])
    return ' '.join(arr) 
class Randomizer(object):

    def __init__(self):
        self.sme = SmilesEnumerator()
    
    def __call__(self, sm):
        sm_r = self.sme.randomize_smiles(sm) # Random transoform
        if sm_r is None:
            sm_spaced = split(sm) # Spacing
        else:
            sm_spaced = split(sm_r) # Spacing
        sm_split = sm_spaced.split()
        if len(sm_split)<=MAX_LEN - 2:
            return sm_split # List
        else:
            return split(sm).split()

    def random_transform(self, sm):
        '''
        function: Random transformation for SMILES. It may take some time.
        input: A SMILES
        output: A randomized SMILES
        '''
        return self.sme.randomize_smiles(sm)

class Seq2seqDataset(Dataset):

    def __init__(self, smiles, vocab, seq_len=220, transform=Randomizer()):
        self.smiles = smiles
        self.vocab = vocab
        self.seq_len = seq_len
        self.transform = transform

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, item):
        sm = self.smiles[item]
        sm = self.transform(sm) # List
        content = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in sm]
        X = [self.vocab.sos_index] + content + [self.vocab.eos_index]
        padding = [self.vocab.pad_index]*(self.seq_len - len(X))
        X.extend(padding)
        return torch.tensor(X)
