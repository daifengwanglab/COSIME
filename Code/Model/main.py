# python Code/Model/main.py --config Code/Model/COSIME.yml > COSIME.log 2>&1
"""COSIME training from a YAML file: single or multiple views, continuous or binary outcome."""
import argparse
import time
from config import resolve,get_config
from train import run

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    cfg=get_config(resolve(args.config))
    run(cfg)
if __name__=='__main__':
    _start = time.perf_counter()
    try:
        main()
    finally:
        _elapsed = time.perf_counter() - _start
        print(f'COSIME total run time: {_elapsed:.2f} seconds ({_elapsed / 60:.2f} minutes; {_elapsed / 3600:.2f} hours)', flush=True)
