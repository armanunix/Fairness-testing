#!/bin/bash
# RQ1 experiments
python3 DICE_Search.py -dataset=bank -sensitive_index=1 -timeout=600 -RQ=1
python3 DICE_Search.py -dataset=census -sensitive_index=9,8,1 -timeout=3600 -RQ=1
python3 DICE_Search.py -dataset=compas -sensitive_index=3,2,1 -timeout=3600 -RQ=1
python3 DICE_Search.py -dataset=credit -sensitive_index=13,9 -timeout=3600 -RQ=1
python3 DICE_Search.py -dataset=default -sensitive_index=5,2 -timeout=3600 -RQ=1
python3 DICE_Search.py -dataset=diabetes -sensitive_index=8 -timeout=3600 -RQ=1
python3 DICE_Search.py -dataset=heart -sensitive_index=2,1 -timeout=3600 -RQ=1
python3 DICE_Search.py -dataset=meps15 -sensitive_index=10,2,1 -timeout=3600 -RQ=1
python3 DICE_Search.py -dataset=meps16 -sensitive_index=10,2,1 -timeout=3600 -RQ=1
python3 DICE_Search.py -dataset=students -sensitive_index=3,2 -timeout=3600 -RQ=1
# RQ2 experiments
python3 DICE_Search.py -dataset=bank -sensitive_index=1 -timeout=900 -RQ=2
python3 DICE_Search.py -dataset=census -sensitive_index=9 -timeout=900 -RQ=2
python3 DICE_Search.py -dataset=census -sensitive_index=8 -timeout=900 -RQ=2
python3 DICE_Search.py -dataset=census -sensitive_index=1 -timeout=900 -RQ=2
python3 DICE_Search.py -dataset=compas -sensitive_index=3 -timeout=900 -RQ=2
python3 DICE_Search.py -dataset=compas -sensitive_index=2 -timeout=900 -RQ=2
python3 DICE_Search.py -dataset=compas -sensitive_index=1 -timeout=900 -RQ=2
python3 DICE_Search.py -dataset=credit -sensitive_index=13 -timeout=900 -RQ=2
python3 DICE_Search.py -dataset=credit -sensitive_index=9 -timeout=900 -RQ=2
python3 DICE_Search.py -dataset=default -sensitive_index=5 -timeout=900 -RQ=2
python3 DICE_Search.py -dataset=default -sensitive_index=2 -timeout=900 -RQ=2
python3 DICE_Search.py -dataset=diabetes -sensitive_index=8 -timeout=900 -RQ=2
python3 DICE_Search.py -dataset=heart -sensitive_index=2 -timeout=900 -RQ=2
python3 DICE_Search.py -dataset=heart -sensitive_index=1 -timeout=900 -RQ=2
python3 DICE_Search.py -dataset=meps15 -sensitive_index=10 -timeout=900 -RQ=2
python3 DICE_Search.py -dataset=meps15 -sensitive_index=2 -timeout=900 -RQ=2
python3 DICE_Search.py -dataset=meps15 -sensitive_index=1 -timeout=900 -RQ=2
python3 DICE_Search.py -dataset=meps16 -sensitive_index=10 -timeout=900 -RQ=2
python3 DICE_Search.py -dataset=meps16 -sensitive_index=2 -timeout=900 -RQ=2
python3 DICE_Search.py -dataset=meps16 -sensitive_index=1 -timeout=900 -RQ=2
python3 DICE_Search.py -dataset=students -sensitive_index=3 -timeout=900 -RQ=2
python3 DICE_Search.py -dataset=students -sensitive_index=2 -timeout=900 -RQ=2
# RQ3 experiments
python3 DICE_Debugging.py -dataset=bank -sensitive_index=1 -num_samples=1000
python3 DICE_Debugging.py -dataset=census -sensitive_index=9,8,1 -num_samples=1000
python3 DICE_Debugging.py -dataset=compas -sensitive_index=3,2,1 -num_samples=1000
python3 DICE_Debugging.py -dataset=credit -sensitive_index=13,9 -num_samples=1000
python3 DICE_Debugging.py -dataset=default -sensitive_index=5,2 -num_samples=1000
python3 DICE_Debugging.py -dataset=diabetes -sensitive_index=8 -num_samples=1000
python3 DICE_Debugging.py -dataset=heart -sensitive_index=2,1 -num_samples=1000
python3 DICE_Debugging.py -dataset=meps15 -sensitive_index=10,2,1 -num_samples=1000
python3 DICE_Debugging.py -dataset=meps16 -sensitive_index=10,2,1 -num_samples=1000
python3 DICE_Debugging.py -dataset=students -sensitive_index=3,2 -num_samples=1000
