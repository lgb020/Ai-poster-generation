import os
from dataset.joint_dataset import get_loader
from joint_solver import Solver
mainroot = os.path.dirname(os.path.realpath(__file__)) + '/'

def main():
    test_fold=mainroot + '/mask/'
    test_loader = get_loader()
    if not os.path.exists(test_fold): os.mkdir(test_fold)
    test = Solver(None, test_loader)
    test.test(test_mode=1)
