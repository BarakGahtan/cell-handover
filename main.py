import pandas as pd

def cells_reader():
    opencell_id_data = pd.read_csv("425.csv")
    obj = pd.read_pickle(r'/Users/barakgahtan/PycharmProjects/Operator-Selector/pickle_rick.pkl')
    x=5

if __name__ == "__main__":
    cells_reader()