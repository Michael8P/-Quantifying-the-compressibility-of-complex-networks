# -*- coding: utf-8 -*-
"""
"""
import networkx as nx
import pandas as pd
import csv
# Read the edgelist from a CSV file where each row represents an interaction between two proteins
edgelist = pd.read_csv("314315.csv")
# Determine the list of unique proteins in the edgelist
print(edgelist)

# Reads edgelist from csv
edge_tuples = []
with open('314315.csv', 'r') as file:
    csv_reader = csv.reader(file)
    # Iterate through each row in the CSV file
    for row in csv_reader:
        # Assuming each row contains two nodes representing an edge
        edge = tuple(row)  # Convert the row values to integers and create a tuple
        edge_tuples.append(edge)  # Append the tuple to the list

# Display the list of tuples
print(edge_tuples)
# create a graph from edgelist
G = nx.from_edgelist(edge_tuples)
print(G)

# converts graph into an undirecteed adjacency matrix
adj_matrix = nx.to_numpy_matrix(G, nodelist=G.nodes())
print(adj_matrix)
