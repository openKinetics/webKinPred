import torch
from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
from utils import split
import json
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc
from sklearn import metrics
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import random
import pickle
import math


def smiles_to_vec(Smiles):
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    mask_index = 4
    vocab = WordVocab.load_vocab('vocab.pkl')
    def get_inputs(sm):
        seq_len = 220
        sm = sm.split()
        if len(sm)>218:
            print('SMILES is too long ({:d})'.format(len(sm)))
            sm = sm[:109]+sm[-109:]
        ids = [vocab.stoi.get(token, unk_index) for token in sm]
        ids = [sos_index] + ids + [eos_index]
        seg = [1]*len(ids)
        padding = [pad_index]*(seq_len - len(ids))
        ids.extend(padding), seg.extend(padding)
        return ids, seg
    def get_array(smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a,b = get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)
        return torch.tensor(x_id), torch.tensor(x_seg)
    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm.load_state_dict(torch.load('trfm_12_23000.pkl'))
    trfm.eval()
    x_split = [split(sm) for sm in Smiles]
    xid, xseg = get_array(x_split)
    X = trfm.encode(torch.t(xid))
    return X


def Seq_to_vec(Sequence):
    for i in range(len(Sequence)):
        if len(Sequence[i]) > 1000:
            Sequence[i] = Sequence[i][:500] + Sequence[i][-500:]
    sequences_Example = []
    for i in range(len(Sequence)):
        zj = ''
        for j in range(len(Sequence[i]) - 1):
            zj += Sequence[i][j] + ' '
        zj += Sequence[i][-1]
        sequences_Example.append(zj)
    tokenizer = T5Tokenizer.from_pretrained("prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("prot_t5_xl_uniref50")
    gc.collect()
    print(torch.cuda.is_available())
    # 'cuda:0' if torch.cuda.is_available() else
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    features = []
    for i in range(len(sequences_Example)):
        print('For sequence ', str(i+1))
        sequences_Example_i = sequences_Example[i]
        sequences_Example_i = [re.sub(r"[UZOB]", "X", sequences_Example_i)]
        ids = tokenizer.batch_encode_plus(sequences_Example_i, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len - 1]
            features.append(seq_emd)
    features_normalize = np.zeros([len(features), len(features[0][0])], dtype=float)
    for i in range(len(features)):
        for k in range(len(features[0][0])):
            for j in range(len(features[i])):
                features_normalize[i][k] += features[i][j][k]
            features_normalize[i][k] /= len(features[i])
    return features_normalize


def Kcat_predict(feature, pH, sequence, smiles, Label):
    # Generate index
    Train_Validation_index = random.sample(range(len(feature)), int(len(feature)*0.8))
    Test_index = []
    for i in range(len(feature)):
        if i not in Train_Validation_index:
            Test_index.append(i)
    Validation_index = random.sample(Train_Validation_index, int(len(Train_Validation_index)*0.2))
    Train_index = []
    for i in range(len(feature)):
        if i not in Validation_index and i not in Test_index:
            Train_index.append(i)
    print(len(Train_index), len(Validation_index), len(Test_index))
    Training_Validation_Test = []
    for i in range(len(feature)):
        if i in Train_index:
            Training_Validation_Test.append(0)
        elif i in Validation_index:
            Training_Validation_Test.append(1)
        else:
            Training_Validation_Test.append(2)
    Train_index = np.array(Train_index)
    Validation_index = np.array(Validation_index)
    Test_index = np.array(Test_index)
    print(Train_index.shape, Validation_index.shape, Test_index.shape)
    # First model
    print(feature[Train_index].shape, pH[Train_index].shape)
    model_1_input = np.concatenate((feature[Train_index], pH[Train_index]), axis=1)
    model_first = ExtraTreesRegressor()
    model_first.fit(model_1_input, Label[Train_index])
    # Second model
    with open("PreKcat_new/0_model.pkl", "rb") as f:
        model_base = pickle.load(f)
    Kcat_baseline = model_base.predict(feature[Validation_index]).reshape([len(Validation_index), 1])
    model_1_2_input = np.concatenate((feature[Validation_index], pH[Validation_index]), axis=1)
    Kcat_calibrated = model_first.predict(model_1_2_input).reshape([len(Validation_index), 1])
    kcat_fused = np.concatenate((Kcat_baseline, Kcat_calibrated), axis=1)
    model_second = LinearRegression()
    model_second.fit(kcat_fused, Label[Validation_index])
    # Final prediction
    model_1_3_input = np.concatenate((feature, pH), axis=1)
    Kcat_calibrated_3 = model_first.predict(model_1_3_input).reshape([len(feature), 1])
    Kcat_baseline_3 = model_base.predict(feature).reshape([len(feature), 1])
    kcat_fused_3 = np.concatenate((Kcat_baseline_3, Kcat_calibrated_3), axis=1)
    Predicted_value = model_second.predict(kcat_fused_3).reshape([len(feature)])
    Training_Validation_Test = np.array(Training_Validation_Test).reshape([len(feature)])
    pH = np.array(pH).reshape([len(Label)])
    Kcat_baseline_3 = np.array(Kcat_baseline_3).reshape([len(feature)])
    Kcat_calibrated_3 = np.array(Kcat_calibrated_3).reshape([len(feature)])
    print(Training_Validation_Test.shape)
    # save
    res = pd.DataFrame({'Value': Label,
                        'sequence': sequence,
                        'smiles': smiles,
                        'pH': pH,
                        'Prediction_first_base': Kcat_baseline_3,
                        'Prediction_first_pH': Kcat_calibrated_3,
                        'Prediction_second': Predicted_value,
                        'Training_Validation_Test': Training_Validation_Test})
    res.to_excel('degree/s2_degree_Kcat.xlsx')


if __name__ == '__main__':
    # Dataset Load
    database = np.array(pd.read_excel('degree/Generated_degree_unified_smiles_572.xlsx')).T
    sequence = database[1]
    smiles = database[3]
    pH = database[5]
    Label = database[4]
    for i in range(len(Label)):
        Label[i] = math.log(Label[i], 10)
    print(max(Label), min(Label))
    pH = np.array(pH).reshape([len(Label), 1])
    with open("degree/features_572_degree_PreKcat.pkl", "rb") as f:
        feature = pickle.load(f)
    # Modelling
    Kcat_predict(feature, pH, sequence, smiles, Label)
