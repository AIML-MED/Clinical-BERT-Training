CUDA_VISIBLE_DEVICES=2,3 python run_bert_inference.py -c configs/bert.json 2>&1 | tee inference.log

