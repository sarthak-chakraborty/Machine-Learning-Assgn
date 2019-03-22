import numpy as np 
import pandas as pd 
import networkx as nx 


df = pd.read_csv("../AAAI.csv")
topics = [df.iloc[i,2].split('\n')for i in range(len(df))]

G = nx.Graph()