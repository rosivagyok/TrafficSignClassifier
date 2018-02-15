import pickle
import os

path = os.path.dirname(__file__)

# load resized (32by32by3), pickled RGB German Traffic Signs Benchmark Database (retrieved from http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
def load_traffic_sign_data(training_file, testing_file):
    with open(path + training_file, mode='rb') as f: # rb -> read data in text mode
        train = pickle.load(f)
    with open(path + testing_file, mode='rb') as f: # rb -> read data in text mode
        test = pickle.load(f)
    return train, test