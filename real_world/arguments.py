import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Real world recommendation experiments')
    parser.add_argument('--dataset', type=str, default='coat', 
                        choices=['coat', 'yahoo', 'kuai'],
                        help='Dataset to use')
    args = parser.parse_args()
    return args