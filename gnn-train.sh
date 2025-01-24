#!/bin/bash

python train_gnn_pipeline.py --dataset DBLP
python train_gnn_pipeline.py --dataset Physics
python train_gnn_pipeline.py --dataset Citeseer_p
python train_gnn_pipeline.py --dataset Cora_p
python train_gnn_pipeline.py --dataset PubMed