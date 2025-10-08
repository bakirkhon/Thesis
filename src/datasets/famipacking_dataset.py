import os
import pathlib

import torch
from torch.utils.data import random_split
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url # InMemoryDataset: for efficient data handling

from datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos

# downloads, processes & saves the dataset
class FamipackingGraphDataset(InMemoryDataset):
    def __init__(self, dataset_name, split, root, transform=None, pre_transform=None,pre_filter=None):
        self.famipacking_file='famipacking.pt' 
        self.dataset_name=dataset_name
        self.split=split
        self.num_graphs=100
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices=torch.load(self.processed_paths[0])

# which raw files should exist before processing:
    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']
    
    @property
    def processed_file_names(self):
        return [self.split+'.pt']
    
    def download(self):
        if self.dataset_name=='famipacking':
            raw_url='https://github.com/bakirkhon/Thesis/raw/main/3D-bin-packing-master/dataset/training_dataset_small.pt'
        else:
            raise ValueError(f'Unknown dataset {self.dataset_name}')
        file_path=download_url(raw_url, self.raw_dir)

        all_graphs=torch.load(file_path)

        g_cpu=torch.Generator()
        g_cpu.manual_seed(0) # sets a deterministic random generator for reproducible splitting

        test_len=int(round(self.num_graphs*0.2))
        train_len=int(round((self.num_graphs-test_len)*0.8))
        val_len=self.num_graphs-train_len-test_len
        indices=torch.randperm(self.num_graphs, generator=g_cpu)
        print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
        train_indices=indices[:train_len]
        val_indices=indices[train_len:train_len+val_len]
        test_indices=indices[train_len+val_len:]

        train_data=[]
        val_data=[]
        test_data=[]

        #converts the arrays into tensors
        for i, graph in enumerate(all_graphs):
            graph['X'] = torch.tensor(graph['X'], dtype=torch.float)
            graph['E'] = torch.tensor(graph['E'], dtype=torch.float)
            if i in train_indices:
                train_data.append(graph)
            elif i in val_indices:
                val_data.append(graph)
            elif i in test_indices:
                test_data.append(graph)
            else:
                raise ValueError(f'Index {i} not in any split')
        
        #save the raw files
        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])

    def process(self):
        file_idx = {'train': 0, 'val': 1, 'test': 2}
        raw_dataset = torch.load(self.raw_paths[file_idx[self.split]])

        data_list=[]
        for graph in raw_dataset:
            X=graph['X']
            E=graph['E']
            n=X.shape[0]
            y=torch.zeros([1, 0]).float() # empty placeholder label for each graph
            # first row=source nodes, second row=destination rows
            edge_index, _=torch_geometric.utils.dense_to_sparse((E.sum(-1)>0).float())
            print(edge_index)
            edge_attr=E[edge_index[0],edge_index[1],:]
            print(edge_attr)
            num_nodes=n*torch.ones(1,dtype=torch.long)
            data=torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=y, n_nodes=num_nodes)
            
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            data_list.append(data)
        #stacks everything into one big tensor storage
        torch.save(self.collate(data_list), self.processed_paths[0])

# creates DataModule object that wraps three datasets
class FamipackingGraphDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs=100):
        self.cfg=cfg
        self.datadir=cfg.dataset.datadir

        # define the root directory where dataset will be stored
        base_path=pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path=os.path.join(base_path, self.datadir)

        datasets = {'train': FamipackingGraphDataset(dataset_name=self.cfg.dataset.name,
                                                     split='train', root=root_path),
                    'val': FamipackingGraphDataset(dataset_name=self.cfg.dataset.name,
                                                     split='val', root=root_path),
                    'test': FamipackingGraphDataset(dataset_name=self.cfg.dataset.name,
                                                     split='test', root=root_path)}
                
        print("train first graph's nodes: ", datasets['train'][0].x)
        print("train first edge pair (source, target): ", datasets['train'][0].edge_index[:, 0])
        print("train first edge feature: ", datasets['train'][0].edge_attr[0])
        print("train target (graph-level) y: ", datasets['train'][0].y)            
        print("train first number of nodes: ", datasets['train'][0].n_nodes)                   

        super().__init__(cfg, datasets)
        self.inner=self.train_dataset

    def __getitem__(self, item):
        return self.inner[item] # datamodule[i] will return the i-th graph from the training set
    
# collects dataset info
class FamipackingDatasetInfo(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule=datamodule
        self.name='famipacking_graphs'
        self.n_nodes=self.datamodule.node_counts()
        print("Distribution of node counts: ", self.n_nodes)
        self.node_types=torch.tensor([1]) # neglecting the node types             
        self.edge_types=self.datamodule.edge_counts()
        print("Distribtution of edge types: ", self.edge_types)
        super().complete_infos(self.n_nodes, self.node_types)

