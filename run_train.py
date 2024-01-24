from src.scripts import *
import os.path as osp

if __name__ == '__main__':
    runtime_path = osp.dirname(osp.abspath(__file__))
    train(runtime_path)