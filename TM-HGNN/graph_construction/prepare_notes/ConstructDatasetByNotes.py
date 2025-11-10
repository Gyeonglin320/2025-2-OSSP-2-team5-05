# ConstructDatasetByNotes.py

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import networkx as nx
import numpy as np
from scipy import sparse
import torch, sys
from torch_geometric.data import Data
pd.set_option('display.max_columns', None)
import os.path as osp
import os
from tqdm import tqdm
from gensim.models import Word2Vec

def graph_to_torch_sparse_tensor(G_true, node_attr=None):
    G = nx.convert_node_labels_to_integers(G_true)
    A_G = np.array(nx.adjacency_matrix(G, weight='edge_type').todense())
    # """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse.csr_matrix(A_G).tocoo()
    edge_index = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col))).to(torch.long)
    edge_attrs = torch.from_numpy(sparse_mx.data).to(torch.float32)

    x = []
    batch_n = []
    batch_t = []
    for node in range(len(G)):
        x.append(G.nodes[node]['node_emb'])
        if node_attr != None:
            for attr in node_attr:
                if attr == 'note_id':
                    batch_n.append(G.nodes[node][attr])
                elif attr == 'cat_id':
                    batch_t.append(G.nodes[node][attr])

    x = torch.from_numpy(np.array(x)).to(torch.float32)
    batch_n = torch.from_numpy(np.array(batch_n)).to(torch.long)
    batch_t = torch.from_numpy(np.array(batch_t)).to(torch.long)

    return edge_index, edge_attrs, x, batch_t, batch_n

def generate_patient_graph(df):
    # print('\nraw---df: ', len(df))
    result_df = combine_same_word_pair(df, col_name='global_freq')   
    result_df['edge_attr'] = 1                                         
    result_graph = nx.from_pandas_edgelist(result_df, 'word1', 'word2', 'edge_attr')
 
    ### remove nan nodes ###
    remove_list = []
    for node in result_graph:
        if node != node:
            remove_list.append(node)
           # result_graph.remove_node(node)
        elif str(node) == 'nan':
            remove_list.append(node)
           # result_graph.remove_node(node)
        else:
            continue
            
    if len(remove_list) > 0 :
        for rm_node in remove_list:
            result_graph.remove_node(rm_node)
            
    return result_graph

# def load_word2vec_embeddings():
#     # word2vec pretrained embeddings
#     w2v_path = './data/DATA_RAW/root/word2vec_100'
#     print('import pretrained word representation from {}...'.format(w2v_path))
#     word_embeddings = Word2Vec.load(osp.join(w2v_path)).wv
#     return word_embeddings

def load_token_embeddings(tokenizer='clinicalbert'):
    tokenizer = tokenizer.lower()
    """
    Load pretrained token embeddings.
    Default: clinicalBERT (.npy)
    Options:
      - tokenizer='clinicalbert' → loads ./data/DATA_RAW/root/clinicalbert_768.npy
      - tokenizer='word2vec'     → loads ./data/DATA_RAW/root/word2vec_100
    Returns:
      get_vec(word:str) -> np.ndarray[emb_dim]
      emb_dim (int)
    """

    # EMB_ROOT = './data/DATA_RAW/root'

    # clinicalBERT 
    # if tokenizer == 'clinicalbert':
    #     emb_path = osp.join(EMB_ROOT, 'clinicalbert_100.npy')
    #     print(f'[INFO] Loading pretrained clinicalBERT embeddings from {emb_path} ...')
    #     emb_dict = np.load(emb_path, allow_pickle=True).item()
    #     emb_dim = 768

    #     def _get_vec(w):
    #         v = emb_dict.get(w)
    #         if v is None:
    #             return np.zeros(emb_dim, dtype=np.float32)
    #         return v.astype(np.float32)

    #     return _get_vec, emb_dim
        # ✅ ClinicalBERT 분기 (cache/note_embeddings에서 로드)

    # if 'clinicalbert' in tokenizer:
    #     print("[INFO] Using ClinicalBERT note embeddings from cache/note_embeddings/")
    #     emb_dim = 768

    #     def get_vec(note_key):
    #         npz_path = osp.join('cache', 'note_embeddings', f'{note_key}.npz')
    #         if osp.exists(npz_path):
    #             try:
    #                 return np.load(npz_path)['emb']
    #             except Exception:
    #                 return np.zeros(emb_dim, dtype=np.float32)
    #         return np.zeros(emb_dim, dtype=np.float32)

    #     return get_vec, emb_dim
    if 'clinicalbert' in tokenizer:
        print("[INFO] Using ClinicalBERT note embeddings from cache/note_embeddings/")
        emb_dim = 768

        def get_vec(note_key):
            import re
            forbidden = {'CON','PRN','AUX','NUL',
                        'COM1','COM2','COM3','COM4','COM5','COM6','COM7','COM8','COM9',
                        'LPT1','LPT2','LPT3','LPT4','LPT5','LPT6','LPT7','LPT8','LPT9'}
            if any(note_key.upper().startswith(x) for x in forbidden):
                print(f"[SKIP-WIN] Reserved filename detected: {note_key}")
                return np.zeros(emb_dim, dtype=np.float32)

            npz_path = osp.join('cache', 'note_embeddings', f'{note_key}.npz')
            if osp.exists(npz_path):
                try:
                    return np.load(npz_path)['emb']
                except Exception as e:
                    print(f"[WARN] Skipping broken embedding file: {npz_path} ({e})")
                    return np.zeros(emb_dim, dtype=np.float32)
            return np.zeros(emb_dim, dtype=np.float32)

        return get_vec, emb_dim

    # word2vec (백워드 호환)
    # w2v_path = osp.join(EMB_ROOT, 'word2vec_100')
    # print(f'[INFO] Loading pretrained word2vec embeddings from {w2v_path} ...')
    # w2v = Word2Vec.load(w2v_path)
    # emb_dim = w2v.vector_size

    # def _get_vec(w):
    #     if w in w2v.wv.key_to_index:
    #         return w2v.wv[w].astype(np.float32)
    #     return np.zeros(emb_dim, dtype=np.float32)

    # return _get_vec, emb_dim
    # ✅ Word2Vec (원래 코드 그대로 유지)
    EMB_ROOT = './data/raw'
    emb_path = osp.join(EMB_ROOT, f'{tokenizer}_100.npy')
    if osp.exists(emb_path):
        emb_dict = np.load(emb_path, allow_pickle=True).item()
        get_vec = lambda w: emb_dict[w] if w in emb_dict else np.random.normal(size=(100,))
        return get_vec, 100


class ConstructDatasetByNotes():
    def __init__(self, pre_path, split, dictionary, task, tokenizer='clinicalBERT'):
        self.pre_path = pre_path
        self.split = split
        self.dictionary = dictionary
        self.task = task    
        self.tokenizer = tokenizer  # 'word2vec' or 'clinicalbert'    
        super(ConstructDatasetByNotes).__init__()
        self.labels = self.get_labels(split)          
        self.cat_path = osp.join(self.pre_path, 'categories.txt')
        # self.all_cats = [a.strip() for a in open(self.cat_path).readlines()]   # Nutrition ~ Respiratory
        
        '''
        categories.txt: (total 14)
       
        Nutrition
        ECG
        Rehab Services
        Case Management 
        Echo
        Pharmacy
        Physician 
        Nursing
        Consult
        General
        Nursing/other
        Radiology
        Social Work
        Discharge summary
        Respiratory 
        '''
        
    def get_labels(self, split):
        # label_patients = pd.read_csv(osp.join(self.pre_path, self.task, split+'_hyper', 'listfile.csv'), sep=',', header=0)
        label_patients = pd.read_csv(osp.join(self.pre_path, 'in-hospital-mortality', split+'_hyper', 'listfile.csv'))
        label_patients['name'] = label_patients.apply(lambda x: str(x['patient']) + '_' + str(x['episode']), axis=1)
        label_patients = label_patients.loc[:, ['name', 'y_true']]
        return label_patients          

    # def make_embedding(self, G, node, node_type):
        # emb = np.zeros(104)
    def make_embedding(self, G, node, node_type, emb_dim):
        emb = np.zeros(4 + emb_dim, dtype=np.float32)
        '''
        0: node_type -> {0:word, 1:note; 2:taxonomy}
        1: word_id 
        2: note_id 
        3: taxonomy_id 
        4:~: for embedding. (word embedding initialized by word2vec)
        '''
        if node_type == 'word':
            emb[0] = 0
            emb[1] = -1
            emb[2] = G.nodes[node]['note_id']
            emb[3] = G.nodes[node]['cat_id']
        elif node_type == 'note':
            emb[0] = 1
            emb[1] = -1
            emb[2] = G.nodes[node]['note_id']
            emb[3] = G.nodes[node]['cat_id']
        elif node_type == 'tax':
            emb[0] = 2
            emb[1] = -1
            emb[2] = -1
            emb[3] = G.nodes[node]['cat_id']
        return emb
    
    def create_all_cats(self, path):
        all_cats = []
        for split in ['train', 'test']:
            # hyper_path = osp.join(self.pre_path, self.task, split+'_hyper')
            hyper_path = osp.join(self.pre_path, self.task, self.split + '_hyper')
            patients = list(filter(lambda x: x in os.listdir(hyper_path), list(self.labels['name'])))  
            for patient in tqdm(patients[:], desc='Iterating over patients in {}_hyper'.format(split)): 
                p_df = pd.read_csv(osp.join(hyper_path, patient), sep='\t', header=0)
                all_cats += p_df['CATEGORY'].tolist()
        all_cats = list(set(all_cats))
        # f = open(osp.join(path, self.task, 'categories.txt'), 'w')
        f = open(osp.join(path, 'categories.txt'), 'w')
        f.write('\n'.join(all_cats))
        f.close()

    def set_node_embedding(self, G, node_attr='node_emb', get_vec=None, emb_dim=768):
        """
        노드 임베딩 설정 함수 (clinicalBERT 기본)
        Args:
            G : networkx graph
            node_attr : 임베딩 속성명 ('node_emb')
            get_vec : 단어를 입력받아 임베딩 벡터(np.ndarray[emb_dim]) 반환하는 함수
            emb_dim : 임베딩 차원 (기본 768)
        """
        for node in G:
            node = str(node)

            # === 노드 타입에 따른 임베딩 초기화 ===
            if node_attr == 'node_emb':
                # 노트 노드 (n_으로 시작)
                if 'n_' in node:
                    emb = self.make_embedding(G, node, node_type='note', emb_dim=emb_dim)
                # 단어 노드
                elif not node.startswith('t_'):
                    emb = self.make_embedding(G, node, node_type='word', emb_dim=emb_dim)

                    # dictionary에서 단어 인덱스 찾기 (없으면 -1)
                    try:
                        emb[1] = self.dictionary.index(node)
                    except ValueError:
                        emb[1] = -1

                    # 단어 벡터 설정
                    if get_vec is not None:
                        emb[4:] = get_vec(node)
                    else:
                        emb[4:] = np.zeros(emb_dim, dtype=np.float32)
                # taxonomy 노드
                else:
                    emb = self.make_embedding(G, node, node_type='tax', emb_dim=emb_dim)

            elif node_attr == 'pe':
                # positional encoding 등 다른 노드 속성을 사용하는 경우
                emb = node_attr[node_attr[:, 0] == node, 1:][0]
                assert (emb.astype(np.float32) == 1).sum() > 0

            else:
                raise ValueError('unknown node attribute')

            # === 노드 속성 저장 ===
            G.nodes[node][node_attr] = emb

        return G


### HyperGraph ###
    def construct_hypergraph_datalist(self):
        print(f"\n[DEBUG] self.tokenizer = {self.tokenizer}")  # ✅ 추가
        print()
        print("<<Start Construct Hypergraph Datalist>>")
        get_vec, emb_dim = load_token_embeddings(self.tokenizer)

        hyper_path = osp.join(self.pre_path + '/' + self.task + '/', self.split + '_hyper/')
        # patients = list(filter(lambda x: x in os.listdir(hyper_path), list(self.labels['name'])))  

        all_files = {osp.splitext(f)[0] for f in os.listdir(hyper_path) if f.endswith('.csv')}
        patients = [n for n in self.labels['name'] if n in all_files]
        
        list_all_cats = [a.strip() for a in open(self.cat_path).readlines()]   # Nutrition ~ Respiratory
        # episode file names into list
        print('<Patient list generation done>')
        data_list = []
        for patient in tqdm(patients[:], desc='Iterating over patients in {}_hyper'.format(self.split)):
            # p_df = pd.read_csv(osp.join(hyper_path, patient), sep='\t', header=0) 
            p_df = pd.read_csv(osp.join(hyper_path, f"{patient}.csv"), sep='\t', header=0)
            # col : ['Hours', 'HADM_ID', 'SUBJECT_ID', 'WORD', 'SENT', 'note_id', 'CATEGORY']

            # change into str if word is not str (int or else...)
            # drop NaN
            p_df = p_df.dropna(axis=0)

            ### Use only 6 major categories ###
            p_df.loc[:, 'CATEGORY'] = p_df.CATEGORY.apply(lambda x: x.strip())  

            # if no notes for 6 CATEGORY => continue (skip to next episode)
            if len(p_df[p_df['CATEGORY'].isin(
                    ['Radiology', 'Nursing', 'Nursing/other', 'ECG', 'Echo', 'Physician'])]) == 0:
                continue

            # filter only 6 categories
            p_df = p_df[p_df['CATEGORY'].isin(['Radiology', 'Nursing', 'Nursing/other', 'ECG', 'Echo', 'Physician'])]

            ### Cutoff notes < 30 ###
            if p_df['note_id'].nunique() > 30:
                max_nid = p_df['note_id'].unique()[30]  # 30 th note_id
                p_df = p_df[p_df['note_id'] < max_nid]  # cut off 0~29 (max 30 notes)

            p_df['WORD'] = p_df['WORD'].astype(str)
            p_df['SENT'] = p_df['SENT'].astype(str)
            p_df['note_id'] = p_df['note_id'].astype(str)
            p_df['note_NM'] = "n_" + p_df['note_id']

            # p_df['SENT_NM'] = p_df['SENT'] + '_n_' + p_df['note_id'] + '_' + p_df['CATEGORY']

            y_p = self.labels[self.labels['name'] == patient]['y_true'].values[0]
            y_p = torch.from_numpy(np.array([y_p])).to(torch.long)  # tensor([1]) OR tensor([0])
            G_n_list = []
            y_n_list = []
            hour_n_list = []

            # per note id generate G_n
            for n_id, n_df in p_df.groupby(by='note_NM'):

                # drop NaN
                n_df = n_df.dropna(axis=0)
                n_df = n_df[n_df['WORD'] != 'nan']

                if len(n_df) > 0:
                    ## edge_type == 1 for word-note edge
                    n_df['edge_type'] = 1 # word-note_edge_type = 0
                    
                    # Graph per note
                    G_n = nx.from_pandas_edgelist(n_df, 'WORD', 'note_NM', 'edge_type')
                        
                    ### Cutoff words per note < 300 ###
                    cut_300 = list(G_n.nodes)[300:]
                    G_n.remove_nodes_from(cut_300)

                    # note ids to node attributes
                    cat_id = list_all_cats.index(n_df['CATEGORY'].values[0].rstrip())  # 14 categories 
                    note_id = int(n_df['note_id'].values[0])
                    attrs = {}
                    for node in G_n:
                        attrs[node] = {'note_id': note_id, 'cat_id': cat_id}
                    nx.set_node_attributes(G_n, attrs)

                    # set node embeddings
                    # G_n = self.set_node_embedding(G_n, node_attr='node_emb', word_embeddings=word_embeddings)
                    G_n = self.set_node_embedding(G_n, node_attr='node_emb', get_vec=get_vec, emb_dim=emb_dim)
                    G_n_list.append(G_n)

                    y_n_list.append([cat_id])
                    # hour_n_list.append([n_df['Hours'].values[0]])
                    hour_n_list.append([0.0])  # dummy hour (not used in TM-HGNN)
                else:
                    continue
            G_n = nx.disjoint_union_all(G_n_list)
            
            for node in range(len(G_n)):
                if G_n.nodes[node]['node_emb'][0] == 0:
                    tax_node = 't_'+ str(G_n.nodes[node]['cat_id'])
                    
                    ## edge_type == 2 for word-taxonomy edge
                    G_n.add_edge(node, tax_node, edge_type=2) # word-taxonomy_edge_type = 1
                    if 'node_emb' not in G_n.nodes[tax_node]:
                        emb = self.make_embedding(G_n, node, node_type='tax', emb_dim=emb_dim)
                        G_n.nodes[tax_node]['node_emb'] = emb # for embedding.
                    if 'note_id' not in G_n.nodes[tax_node]:
                        G_n.nodes[tax_node]['note_id'] = G_n.nodes[node]['note_id']
                        G_n.nodes[tax_node]['cat_id'] = G_n.nodes[node]['cat_id']
                        

            edge_index_n, edge_index_mask, x_n, batch_t, batch_n = graph_to_torch_sparse_tensor(G_n, node_attr=['note_id', 'cat_id'])

            y_n = torch.from_numpy(np.array(y_n_list)).to(torch.long)
            hour_n = torch.from_numpy(np.array(hour_n_list)).to(torch.float32)


            # data = Data(x_n=x_n, edge_index_n=edge_index_n, edge_index_mask=edge_index_mask, hour_n=hour_n, y_n=y_n, y_p=y_p, batch_n=batch_n, batch_t=batch_t)
            # data.num_nodes = x_n.size(0)
            # data.x = data.x_n 
            # data.edge_index = data.edge_index_n
            # assert data.edge_index.dtype == torch.long and data.edge_index.dim() == 2 and data.edge_index.size(0) == 2
            # data.num_nodes = data.x_n.shape[0]
            data = Data(
                x_n=x_n,
                edge_index_n=edge_index_n,
                edge_index_mask=edge_index_mask,
                hour_n=hour_n,
                y_n=y_n,
                y_p=y_p,
                batch_n=batch_n,
                batch_t=batch_t
            )

            # ✅ num_nodes 명시 (tensor형으로)
            data.num_nodes = torch.tensor([x_n.size(0)], dtype=torch.long)

            # ✅ PyG 표준 alias 지정
            data.x = data.x_n
            data.edge_index = data.edge_index_n
            data.y = y_p.float()
            data.edge_mask = data.edge_index_mask


            # ✅ 구조 검증
            assert data.edge_index.dtype == torch.long
            assert data.edge_index.dim() == 2 and data.edge_index.size(0) == 2
            
            data_list.append(data)

        print('<Hypergraph Data list generation done>')
        print('<<End Construct Hypergraph Datalist>>')
        print()

        return data_list


if __name__ == '__main__':
    task = 'in-hospital-mortality'
    raw_path = './data/raw/'
    pre_path = './data/interim'
    dictionary = open(os.path.join(pre_path, 'root', 'vocab.txt')).read().split()
    # cdbn = ConstructDatasetByNotes(pre_path, split='train', dictionary=dictionary)  # split = train. test
    cdbn = ConstructDatasetByNotes(pre_path, split='train', dictionary=dictionary, task=task, tokenizer='clinicalbert')
    # cdbn.create_all_cats(pre_path)
    data_list = cdbn.construct_hypergraph_datalist()