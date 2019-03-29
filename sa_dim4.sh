#!/bin/bash

./optimize_sa.py --dim 4 --mt 2 --dataset ../kimia99/ -o kimia99_random_weight_dim4.pkl 95 0.965 8 6 0 0 0 > output_random_dim4.txt
./optimize_sa.py --dim 4 --mt 2 --dataset ../kimia99/ -o kimia99_si_dim4.pkl 96 0.965 8 6 1.0 0 0 > output_si_dim4.txt
./optimize_sa.py --dim 4 --mt 2 --dataset ../kimia99/ -o kimia99_db_dim4.pkl 96 0.965 8 6 0 1.0 0 > output_db_dim4.txt
./optimize_sa.py --dim 4 --mt 2 --dataset ../kimia99/ -o kimia99_ch_dim4.pkl 96 0.965 8 6 0 0 1.0 > output_ch_dim4.txt
