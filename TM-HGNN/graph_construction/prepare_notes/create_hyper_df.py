from __future__ import absolute_import
from __future__ import print_function
import os
import argparse
import pandas as pd
from tqdm import tqdm

def note_hyper(p_df):
    dfs = []
    note_id = 0

    text_col = 'fixed TEXT' if 'fixed TEXT' in p_df.columns else 'TEXT'

    for i, note in enumerate(p_df[text_col]):
        # hours = p_df.iloc[i, :]['Hours']
        category = p_df.iloc[i, :]['CATEGORY']
        hadm_id = p_df.iloc[i, :]['HADM_ID']
        subject_id = p_df.iloc[i, :]['SUBJECT_ID']
        sents = str(note).split('\n')
        sent_id = 0
        for sent in sents:
            item_list = sent.split(' ')
            for item in item_list:
                dfs.append([hadm_id, subject_id, item, sent_id, note_id, category])
            sent_id += 1
        note_id += 1

    # dfs = pd.DataFrame(dfs, columns=['Hours', 'HADM_ID', 'SUBJECT_ID', 'WORD', 'SENT', 'note_id', 'CATEGORY'])
    dfs = pd.DataFrame(dfs, columns=['HADM_ID', 'SUBJECT_ID', 'WORD', 'SENT', 'note_id', 'CATEGORY'])
    # dfs.append([hadm_id, subject_id, item, sent_id, note_id, category])
    if len(dfs) == 0:
        dfs = ''
    return dfs

def create_hyper_df(partition, action):
    output_dir = os.path.join(args.pre_path, args.task, partition+'_hyper')
    listfile = os.path.join(args.pre_path, args.task, f"{partition}_hyper", "listfile.csv")
    if not os.path.exists(listfile):
        raise FileNotFoundError(f"listfile not found: {listfile}")
    lf = pd.read_csv(listfile)  # columns: patient, episode, y_true

    split = partition + '_note'
    if action == 'make':
        os.makedirs(output_dir, exist_ok=True)
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        #patients = list(filter(lambda x: x.find("episode") != -1, os.listdir(os.path.join(args.raw_path, args.task, split))))
        input_file = os.path.join(args.pre_path, args.task, f"{partition}_note", f"cleaned_notes_{partition}.csv")
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"{input_file} not found")
        
        print(f"Loading {input_file}...")
        p_all = pd.read_csv(input_file)
        gb = p_all.groupby(["SUBJECT_ID", "HADM_ID"], sort=False)
        
        for row in tqdm(lf.itertuples(index=False), total=len(lf), desc=f"make {partition}_hyper"):
            pid, hadm = int(row.patient), int(row.episode)
            name = f"{pid}_{hadm}"
            out_path = os.path.join(output_dir, f"{name}.csv")

            if os.path.exists(out_path):
                continue
            try:
                p_df = gb.get_group((pid, hadm))
            except KeyError:
                continue

            p_hyper_df = note_hyper(p_df)
            if isinstance(p_hyper_df, str) or len(p_hyper_df) == 0:
                continue

            p_hyper_df.to_csv(out_path, sep="\t", index=False)

        print(f"{partition} set complete! ({len([f for f in os.listdir(output_dir) if f.endswith('.csv')])} files)")

        # # for patient in tqdm(patients[:], desc='Iterating over patients in {}_{}_{}'.format(args.raw_path, args.task, split)):
        # #     p_df = pd.read_csv(os.path.join(args.raw_path, args.task, split, patient), sep=',', header=0)
        
        # if not os.path.exists(input_file):
        #     raise FileNotFoundError(f"{input_file} not found")
        
        # print(f"Loading {input_file}...")
        # p_df = pd.read_csv(input_file)

        # p_hyper_df = note_hyper(p_df)
        # if len(p_hyper_df) > 0:
        #     # out_path = os.path.join(output_dir, f"hyper_{partition}.csv")
        #     # p_hyper_df.to_csv(os.path.join(output_dir, patient), sep='\t', index=False)
        #     # p_hyper_df.to_csv(out_path, sep='\t', index=False)
        #     print(f"Saved: {out_path} (rows={len(p_hyper_df)})")
        # else:
        #     # print('--> Warning: No nodes in patient {}!!'.format(patient))
        #     print(f"Warning: No nodes generated for {partition}!")

        # #print('training sample complete! please check in {}'.format(output_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create hypergraph dataframe for in hospital mortality prediction task.")
    parser.add_argument('--raw_path', type=str, default='./data/raw',
                        help="Directory where the cleaned data should be stored (Input).")
    parser.add_argument('--pre_path', type=str, default='./data/interim',
                        help="Directory where the processed data should be stored (Output).")    
    parser.add_argument('--dimension', type=str, default='100' , help='input dimension')
    parser.add_argument('--task', type=str, default='in-hospital-mortality' , help='task name: [in-hospital-mortality]')
    parser.add_argument('--tokenizer', type=str, default='word2vec')
    parser.add_argument('--window_size', type=int, default=3)
    parser.add_argument('--partition', type=str, default='test')
    parser.add_argument('--action', type=str, default='make')

    args, _ = parser.parse_known_args()

    print('Creating train HyperSamples...')
    create_hyper_df('train', args.action)
    print('Creating val HyperSamples...')       
    create_hyper_df('val', args.action)        
    print('Creating test HyperSamples...')
    create_hyper_df('test', args.action)
