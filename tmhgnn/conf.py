import sys, os
import argparse

def sys_append(a, a_1):
    sys.path.append(os.path.join(a, a_1))

def return_paths(a):
    IMDB_PATH = os.path.join(a, './data/IMDB_HCUT')
    PRE_PATH = os.path.join(a, './data/DATA_PRE')
    RAW_PATH = os.path.join(a, './data/DATA_RAW')
    return IMDB_PATH, PRE_PATH, RAW_PATH

"""
def user_config():
    home = "TM-HGNN"
    graph_path = 'graph_construction'
    IMDB_PATH, PRE_PATH, RAW_PATH = return_paths(home)
    sys_append(home, graph_path)_summary_
"""

def user_config():
    # 프로젝트 루트 자동 계산
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
    BASE_DIR = os.path.dirname(BASE_DIR)  # train.py 있는 폴더 기준 상위

    sys.path.append(os.path.join(BASE_DIR, 'graph_construction'))
    
### Configs ###
def parse_arguments(): 
    parser = argparse.ArgumentParser()
    
    ## Define DataLoader & Model
    parser.add_argument('-d', '--dload', default='multi_hyper')
    parser.add_argument('-m', '--model', default='TM_HGNN')
    
    # Output path for ckpts 
    parser.add_argument('-o', '--output', default='./tmhgnn/TM_HGNN.pth')    
    
    # Save results
    parser.add_argument('-e', '--exp-name', default='./tmhgnn/results')    
    
    # task
    parser.add_argument('-t', '--task', default='in-hospital-mortality', type=str)  
    
    # data
    parser.add_argument('--tokenizer', default='clinicalbert', type=str)  
    parser.add_argument('--dtype', default='hyper', type=str)  # Hypergraphs
    
    # training
    parser.add_argument('-b', '--bsz', default=32, type=int) 
    parser.add_argument('--optimizer', default='Adam')          
    parser.add_argument('--init-lr', default=0.001, type=float)     
    parser.add_argument('--weight-decay', default=0, type=float)
    parser.add_argument('--loss', default='BCEWithLogitsLoss')      # Binary Classification           
    parser.add_argument('--num-epochs', default=50, type=int)       
    
    # model
    parser.add_argument('--layers', default=3, type=int)  
    parser.add_argument('--heads-1', default=8, type=int)  
    parser.add_argument('--heads-2', default=8, type=int)

    parser.add_argument('--hidden-channels', default=8, type=int)
    parser.add_argument('--node-features', default=104, type=int)
    
    # seed
    parser.add_argument('--seed', default=50, type=int)

    return parser.parse_args()