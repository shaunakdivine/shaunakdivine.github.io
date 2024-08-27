---
layout: default
title: Author Prediction Reuters
date:   2024-08-26 2:06:00 -0500
categories: machine learning
permalink: /projects/project2/

---

# Author Prediction Using a GCN

This project attempts to predict authors using a graph-based approach as compared to a simple SVM model.

- **Date:** August 2024
- **Technologies Used:** Python, Graph Convolutional Networks, PyTorch, SVM


# Overview of Author Prediction Experiments

### 1. Introduction and Question
The overall question we were trying to answer was could we predict the author of a document using a graph-based approach. To do this we explored creating a 'similarity graph' using different text comparison methods (TF-IDF and BERT) and then predicted authors using a Graph Convolutional Network (GCN). We then compared these results to a very simple Support Vector Machine (SVM) approach to see how the outcomes differed. We worked on altering multiple parts of the similarity checks and graph creation to improve accuracy. The experiments were conducted using the Reuters C50 dataset, consisting of documents from 50 different authors. The detailed steps to re-create our analysis are outlined throughout the report, cell-by-cell. 

### 2. Approach and Methodology

- Data Preparation
    - Each document was represented as a feature vector using the TF-IDF technique, limiting the vocabulary to the top 5,000 words across all documents. The target labels, representing authors, were encoded using LabelEncoder to convert the author names into integers.

A. Graph-Based Approach (GCN)
- For this approach, we wanted to construct the graph based on similarity of the documents, hoping to see that documents written by the same author would be more similar and thus form connected components associated with certain authors. Then, we could pass this structure through the GCN to accurately predict the authors of the test set. 
- Graph Construction:
    - Nodes: Each node represents a document in the dataset.
    - Edges: Edges between nodes were created based on cosine similarity of the TF-IDF vectors. An edge was added if the cosine similarity exceeded a threshold (e.g., 0.5 or 0.7). We also attempted creating edge weights using the cosine similarity of BERT embeddings instead of TF-IDF vectors. 
    - Node Features: Each node was assigned its corresponding TF-IDF vector.
    - Edge Weights: The weights of the edges were set to the cosine similarity between the documents.
- Graph Convolutional Network (GCN):
    - Model: A 2-layer GCN model was trained on the constructed graph. We used the Adam Optimizer and Cross Entropy Loss as the Loss Criterion, with a learning rate of .01
    - Node Features: Each document's TF-IDF vector was used as its feature vector.
    - Output: The model predicts the author of each document.
- Training and Evaluation:
    - Training: The GCN was trained using only the training nodes, while keeping the test nodes masked. We trained on a range of 200-1000 epochs.
    - Evaluation: The model was evaluated on the test set, and we investigated the top-1 and top-5 accuracies (meaning was the author in the top 1 or top 5 of predicted authors)
        - During training and testing, we used confusion matrices and per-author accuracy to evaluate the model

B. Simple SVM
- Model: An SVM classifier with a linear kernel was trained on the TF-IDF vectors.
- Training and Testing: The SVM was trained and tested on the respective provided train and test sets.

Tools and Libraries
    - We implemented the GCN using PyTorch Geometric and used scikit-learn for the SVM. TF-IDF feature extraction was done using TfidfVectorizer from scikit-learn.


### 3. Results
Specific confusion matrix and accuracy plot figures are throughout the document.
We compared the performance of the different models based on two metrics:

- Top-1 Accuracy: The percentage of documents for which the model correctly predicted the author.
- Top-5 Accuracy: The percentage of documents where the true author was among the top 5 predicted authors.

| Model                        | Top-1 Accuracy | Top-5 Accuracy |
|------------------------------|----------------|----------------|
| GCN with threshold = 0.5     | 62.60%         | 94.20%         |
| GCN with threshold = 0.7     | 68.16%         | 96.40%         |
| Simple SVM                   | 65.80%         | 94.00%         |
| Baseline Accuracy            | 2.00%          | 10.00%         |

We also tested using GAT (Graph Attention Network) layers and BERT embeddings, but the top-1 accuracy was below our GCN model. The accuracy of these was **56.32%** and **26.20%** respectively, so we did not continue forward with them. 

Observations:
The GCN model, which leverages both the document features and the graph structure, showed strong performance. The inclusion of edges based on document similarity did indeed improve the model's ability to capture relationships between documents from the same author. However, it predicted poorly on some authors specifically, and we believe this is due to multiple authors having very similar writing subjects. Overall, our GCN outperformed the simple SVM, but this model could be improved; it was included only to provide a reference.


### 4. Conclusion and Analysis

In conclusion, the graph-based GCN model outperformed the simple SVM in this author prediction task. By utilizing both document content and relationships between documents, the GCN was able to more accurately predict authorship, especially when considering top-5 accuracy. It showed strong ability to identify authors at almost 70%. This provides an interesting approach, as more documents could be added to the graph using TF-IDF which should continue to grow its predictive power. In the future, it would be interesting to experiment further with different edge creation or incorporating additional features into the graph (e.g., document metadata). Specifically, trying Word2Vec or Doc2Vec could provide interesting results. I would also like to try ensembling the SVM and the graph-based approach to enhance the overall predictive ability. 




### Validating Data

First step was just making sure the data was migrated correctly. I wanted to make sure there were no missing articles or authors, and that the files were set up correctly. This script simply goes into the folder, checks that there are 50 authors in each of the train and test set, then confirms that each author has 50 articles associated with them. Doesn't check the contents of any files, just the overall structure.


```python
import os

def validate_reuters_c50_structure(data_path):
    c50train_path = os.path.join(data_path, 'C50train')
    c50test_path = os.path.join(data_path, 'C50test')
    if not os.path.exists(c50train_path):
        print(f"Error: '{c50train_path}' does not exist.")
        return False
    if not os.path.exists(c50test_path):
        print(f"Error: '{c50test_path}' does not exist.")
        return False
    
    def validate_author_folders(folder_path):
        authors = os.listdir(folder_path)
        if len(authors) != 50:
            print(f"Error: Expected 50 author folders in '{folder_path}', but found {len(authors)}.")
            return False
        
        for author in authors:
            author_path = os.path.join(folder_path, author)
            if not os.path.isdir(author_path):
                print(f"Error: '{author_path}' is not a directory.")
                return False
            
            articles = os.listdir(author_path)
            if len(articles) != 50:
                print(f"Error: Expected 50 articles in '{author_path}', but found {len(articles)}.")
                return False
        return True
    
    print("Validating C50train...")
    if not validate_author_folders(c50train_path):
        print("C50train validation failed.")
        return False
    else:
        print("C50train validation passed.")
    
    print("Validating C50test...")
    if not validate_author_folders(c50test_path):
        print("C50test validation failed.")
        return False
    else:
        print("C50test validation passed.")
    
    print("Reuters C50 dataset structure is correct.")
    return True

data_path = 'data/ReutersC50'
validate_reuters_c50_structure(data_path)

```

    Validating C50train...
    C50train validation passed.
    Validating C50test...
    C50test validation passed.
    Reuters C50 dataset structure is correct.
    




    True



### Data Loading and Feature Extraction
The Reuters C50 dataset is loaded using a function that reads documents from folders corresponding to 50 authors. Both training and test data are combined, and authors' names are encoded as integer labels using LabelEncoder.

The text of each document is converted into TF-IDF vectors with a maximum of 5,000 features using TfidfVectorizer. These vectors represent the documents and will be used for author prediction in subsequent models.


```python
import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import numpy as np

def load_data(root_dir):
    texts = []
    labels = []
    author_folders = os.listdir(root_dir)
    
    for author in author_folders:
        author_dir = os.path.join(root_dir, author)
        for filename in os.listdir(author_dir):
            with open(os.path.join(author_dir, filename), 'r', encoding='latin-1') as file:
                texts.append(file.read())
                labels.append(author)
    
    return texts, labels

train_texts, train_labels = load_data('data/ReutersC50/C50train')
test_texts, test_labels = load_data('data/ReutersC50/C50test')


all_texts = train_texts + test_texts
all_labels = train_labels + test_labels

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(all_labels)

vectorizer = TfidfVectorizer(max_features=5000)
features = vectorizer.fit_transform(all_texts)
```

### Graph Construction
A graph is created where each node represents a document with TF-IDF features. Edges between nodes are added based on cosine similarity between document feature vectors, with a threshold of 0.5. The process iterates over all document pairs, adding an edge if their similarity exceeds the threshold, and records the time taken for node and edge addition.


```python
import time


G = nx.Graph()


print("Adding nodes to the graph...")
start_time = time.time()

for i in range(features.shape[0]):
    G.add_node(i, feature=features[i], label=labels[i])

end_time = time.time()
print(f"Nodes added: {features.shape[0]} nodes in {end_time - start_time:.2f} seconds.")

## We want to add edges based on cosine similarity
print("Adding edges to the graph based on cosine similarity...")
start_time = time.time()

threshold = 0.5 
total_comparisons = (features.shape[0] * (features.shape[0] - 1)) // 2
comparison_count = 0

for i in range(features.shape[0]):
    for j in range(i + 1, features.shape[0]):
        similarity = cosine_similarity(features[i], features[j])[0][0]
        if similarity > threshold:
            G.add_edge(i, j, weight=similarity)
        comparison_count += 1
        
        ## Just wanted progress updates
        if comparison_count % 100000 == 0:
            elapsed_time = time.time() - start_time
            print(f"Processed {comparison_count}/{total_comparisons} comparisons "
                  f"({(comparison_count / total_comparisons) * 100:.2f}%) "
                  f"in {elapsed_time:.2f} seconds.")

end_time = time.time()
print(f"Edges added in {end_time - start_time:.2f} seconds.")

```

    Adding nodes to the graph...
    Nodes added: 5000 nodes in 0.30 seconds.
    Adding edges to the graph based on cosine similarity...
    Processed 100000/12497500 comparisons (0.80%) in 79.67 seconds.
    Processed 200000/12497500 comparisons (1.60%) in 158.44 seconds.
    Processed 300000/12497500 comparisons (2.40%) in 249.68 seconds.
    Processed 400000/12497500 comparisons (3.20%) in 342.38 seconds.
    Processed 500000/12497500 comparisons (4.00%) in 431.50 seconds.
    Processed 600000/12497500 comparisons (4.80%) in 518.58 seconds.
    Processed 700000/12497500 comparisons (5.60%) in 604.59 seconds.
    Processed 800000/12497500 comparisons (6.40%) in 692.73 seconds.
    Processed 900000/12497500 comparisons (7.20%) in 781.30 seconds.
    Processed 1000000/12497500 comparisons (8.00%) in 873.81 seconds.
    Processed 1100000/12497500 comparisons (8.80%) in 963.62 seconds.
    Processed 1200000/12497500 comparisons (9.60%) in 1056.22 seconds.
    Processed 1300000/12497500 comparisons (10.40%) in 1144.47 seconds.
    Processed 1400000/12497500 comparisons (11.20%) in 1230.62 seconds.
    Processed 1500000/12497500 comparisons (12.00%) in 1336.90 seconds.
    Processed 1600000/12497500 comparisons (12.80%) in 1479.97 seconds.
    Processed 1700000/12497500 comparisons (13.60%) in 1568.05 seconds.
    Processed 1800000/12497500 comparisons (14.40%) in 1643.68 seconds.
    Processed 1900000/12497500 comparisons (15.20%) in 1719.98 seconds.
    Processed 2000000/12497500 comparisons (16.00%) in 1797.91 seconds.
    Processed 2100000/12497500 comparisons (16.80%) in 1874.13 seconds.
    Processed 2200000/12497500 comparisons (17.60%) in 1949.85 seconds.
    Processed 2300000/12497500 comparisons (18.40%) in 2025.82 seconds.
    Processed 2400000/12497500 comparisons (19.20%) in 2096.32 seconds.
    Processed 2500000/12497500 comparisons (20.00%) in 2166.55 seconds.
    Processed 2600000/12497500 comparisons (20.80%) in 2237.80 seconds.
    Processed 2700000/12497500 comparisons (21.60%) in 2308.06 seconds.
    Processed 2800000/12497500 comparisons (22.40%) in 2379.87 seconds.
    Processed 2900000/12497500 comparisons (23.20%) in 2450.57 seconds.
    Processed 3000000/12497500 comparisons (24.00%) in 2520.85 seconds.
    Processed 3100000/12497500 comparisons (24.80%) in 2591.51 seconds.
    Processed 3200000/12497500 comparisons (25.61%) in 2664.67 seconds.
    Processed 3300000/12497500 comparisons (26.41%) in 2734.83 seconds.
    Processed 3400000/12497500 comparisons (27.21%) in 2809.48 seconds.
    Processed 3500000/12497500 comparisons (28.01%) in 2880.92 seconds.
    Processed 3600000/12497500 comparisons (28.81%) in 2952.10 seconds.
    Processed 3700000/12497500 comparisons (29.61%) in 3023.11 seconds.
    Processed 3800000/12497500 comparisons (30.41%) in 3094.97 seconds.
    Processed 3900000/12497500 comparisons (31.21%) in 3168.29 seconds.
    Processed 4000000/12497500 comparisons (32.01%) in 3244.61 seconds.
    Processed 4100000/12497500 comparisons (32.81%) in 3314.36 seconds.
    Processed 4200000/12497500 comparisons (33.61%) in 3384.95 seconds.
    Processed 4300000/12497500 comparisons (34.41%) in 3455.01 seconds.
    Processed 4400000/12497500 comparisons (35.21%) in 3525.50 seconds.
    Processed 4500000/12497500 comparisons (36.01%) in 3595.31 seconds.
    Processed 4600000/12497500 comparisons (36.81%) in 3668.31 seconds.
    Processed 4700000/12497500 comparisons (37.61%) in 3748.97 seconds.
    Processed 4800000/12497500 comparisons (38.41%) in 3827.56 seconds.
    Processed 4900000/12497500 comparisons (39.21%) in 3902.43 seconds.
    Processed 5000000/12497500 comparisons (40.01%) in 3976.03 seconds.
    Processed 5100000/12497500 comparisons (40.81%) in 4052.52 seconds.
    Processed 5200000/12497500 comparisons (41.61%) in 4128.03 seconds.
    Processed 5300000/12497500 comparisons (42.41%) in 4203.78 seconds.
    Processed 5400000/12497500 comparisons (43.21%) in 4275.85 seconds.
    Processed 5500000/12497500 comparisons (44.01%) in 4349.21 seconds.
    Processed 5600000/12497500 comparisons (44.81%) in 4422.78 seconds.
    Processed 5700000/12497500 comparisons (45.61%) in 4493.20 seconds.
    Processed 5800000/12497500 comparisons (46.41%) in 4565.37 seconds.
    Processed 5900000/12497500 comparisons (47.21%) in 4636.54 seconds.
    Processed 6000000/12497500 comparisons (48.01%) in 4707.65 seconds.
    Processed 6100000/12497500 comparisons (48.81%) in 4780.94 seconds.
    Processed 6200000/12497500 comparisons (49.61%) in 4854.87 seconds.
    Processed 6300000/12497500 comparisons (50.41%) in 4929.19 seconds.
    Processed 6400000/12497500 comparisons (51.21%) in 5001.68 seconds.
    Processed 6500000/12497500 comparisons (52.01%) in 5073.12 seconds.
    Processed 6600000/12497500 comparisons (52.81%) in 5147.50 seconds.
    Processed 6700000/12497500 comparisons (53.61%) in 5242.36 seconds.
    Processed 6800000/12497500 comparisons (54.41%) in 5317.08 seconds.
    Processed 6900000/12497500 comparisons (55.21%) in 5390.51 seconds.
    Processed 7000000/12497500 comparisons (56.01%) in 5462.87 seconds.
    Processed 7100000/12497500 comparisons (56.81%) in 5539.12 seconds.
    Processed 7200000/12497500 comparisons (57.61%) in 5615.23 seconds.
    Processed 7300000/12497500 comparisons (58.41%) in 5689.12 seconds.
    Processed 7400000/12497500 comparisons (59.21%) in 5766.66 seconds.
    Processed 7500000/12497500 comparisons (60.01%) in 5844.36 seconds.
    Processed 7600000/12497500 comparisons (60.81%) in 5919.97 seconds.
    Processed 7700000/12497500 comparisons (61.61%) in 5993.75 seconds.
    Processed 7800000/12497500 comparisons (62.41%) in 6071.39 seconds.
    Processed 7900000/12497500 comparisons (63.21%) in 6141.71 seconds.
    Processed 8000000/12497500 comparisons (64.01%) in 6212.64 seconds.
    Processed 8100000/12497500 comparisons (64.81%) in 6290.00 seconds.
    Processed 8200000/12497500 comparisons (65.61%) in 6368.71 seconds.
    Processed 8300000/12497500 comparisons (66.41%) in 6444.63 seconds.
    Processed 8400000/12497500 comparisons (67.21%) in 6519.56 seconds.
    Processed 8500000/12497500 comparisons (68.01%) in 6592.46 seconds.
    Processed 8600000/12497500 comparisons (68.81%) in 6665.15 seconds.
    Processed 8700000/12497500 comparisons (69.61%) in 6747.37 seconds.
    Processed 8800000/12497500 comparisons (70.41%) in 6821.73 seconds.
    Processed 8900000/12497500 comparisons (71.21%) in 6898.75 seconds.
    Processed 9000000/12497500 comparisons (72.01%) in 6974.73 seconds.
    Processed 9100000/12497500 comparisons (72.81%) in 7048.61 seconds.
    Processed 9200000/12497500 comparisons (73.61%) in 7122.11 seconds.
    Processed 9300000/12497500 comparisons (74.41%) in 7295.01 seconds.
    Processed 9400000/12497500 comparisons (75.22%) in 7467.03 seconds.
    Processed 9500000/12497500 comparisons (76.02%) in 7635.28 seconds.
    Processed 9600000/12497500 comparisons (76.82%) in 7733.59 seconds.
    Processed 9700000/12497500 comparisons (77.62%) in 7806.53 seconds.
    Processed 9800000/12497500 comparisons (78.42%) in 7879.04 seconds.
    Processed 9900000/12497500 comparisons (79.22%) in 7950.22 seconds.
    Processed 10000000/12497500 comparisons (80.02%) in 8023.58 seconds.
    Processed 10100000/12497500 comparisons (80.82%) in 8108.01 seconds.
    Processed 10200000/12497500 comparisons (81.62%) in 8195.63 seconds.
    Processed 10300000/12497500 comparisons (82.42%) in 8275.57 seconds.
    Processed 10400000/12497500 comparisons (83.22%) in 8363.00 seconds.
    Processed 10500000/12497500 comparisons (84.02%) in 8451.67 seconds.
    Processed 10600000/12497500 comparisons (84.82%) in 8525.21 seconds.
    Processed 10700000/12497500 comparisons (85.62%) in 8599.56 seconds.
    Processed 10800000/12497500 comparisons (86.42%) in 8671.43 seconds.
    Processed 10900000/12497500 comparisons (87.22%) in 8747.51 seconds.
    Processed 11000000/12497500 comparisons (88.02%) in 8825.90 seconds.
    Processed 11100000/12497500 comparisons (88.82%) in 8903.93 seconds.
    Processed 11200000/12497500 comparisons (89.62%) in 8974.76 seconds.
    Processed 11300000/12497500 comparisons (90.42%) in 9049.03 seconds.
    Processed 11400000/12497500 comparisons (91.22%) in 9122.48 seconds.
    Processed 11500000/12497500 comparisons (92.02%) in 9196.59 seconds.
    Processed 11600000/12497500 comparisons (92.82%) in 9271.60 seconds.
    Processed 11700000/12497500 comparisons (93.62%) in 9343.80 seconds.
    Processed 11800000/12497500 comparisons (94.42%) in 9417.21 seconds.
    Processed 11900000/12497500 comparisons (95.22%) in 9496.29 seconds.
    Processed 12000000/12497500 comparisons (96.02%) in 9570.11 seconds.
    Processed 12100000/12497500 comparisons (96.82%) in 9643.07 seconds.
    Processed 12200000/12497500 comparisons (97.62%) in 9717.63 seconds.
    Processed 12300000/12497500 comparisons (98.42%) in 9791.66 seconds.
    Processed 12400000/12497500 comparisons (99.22%) in 9865.92 seconds.
    Edges added in 9940.02 seconds.
    

### Graph Visualization
The constructed graph is visualized using nx.draw(), where each document (node) is represented with a small size (node_size=10), and labels (author names) are hidden (with_labels=False). The plt.show() command displays the graph plot.




```python

nx.draw(G, node_size=10, with_labels=False)
plt.show()
```


    
![png](assets/reutersCorpus_files/reutersCorpus_8_0.png)
    


### Edge Count and Weights
The total number of edges in the graph is printed, followed by the weights of the first 100 edges. Each edge's weight represents the cosine similarity between the connected documents (nodes).


```python
print(f"Total number of edges in the graph: {G.number_of_edges()}")

print("\nFirst 100 edge weights:")
for i, (u, v, attr) in enumerate(G.edges(data=True)):
    if i < 100:  
        print(f"Edge ({u}, {v}) - Weight: {attr['weight']:.4f}")
    else:
        break
```

    Total number of edges in the graph: 32730
    
    First 100 edge weights:
    Edge (1, 4597) - Weight: 0.5154
    Edge (3, 4) - Weight: 0.9272
    Edge (3, 21) - Weight: 0.6975
    Edge (3, 22) - Weight: 0.6918
    Edge (3, 38) - Weight: 0.5774
    Edge (3, 2518) - Weight: 0.5962
    Edge (3, 2519) - Weight: 0.5913
    Edge (3, 2521) - Weight: 0.5987
    Edge (4, 21) - Weight: 0.6852
    Edge (4, 22) - Weight: 0.6732
    Edge (4, 38) - Weight: 0.5750
    Edge (4, 2518) - Weight: 0.6147
    Edge (4, 2519) - Weight: 0.6057
    Edge (4, 2521) - Weight: 0.6175
    Edge (7, 8) - Weight: 0.6621
    Edge (7, 48) - Weight: 0.6971
    Edge (7, 550) - Weight: 0.5214
    Edge (7, 594) - Weight: 0.5214
    Edge (8, 48) - Weight: 0.6832
    Edge (9, 11) - Weight: 0.9984
    Edge (10, 13) - Weight: 0.5486
    Edge (10, 35) - Weight: 0.5251
    Edge (10, 49) - Weight: 0.5685
    Edge (10, 2072) - Weight: 0.6757
    Edge (10, 2073) - Weight: 0.6051
    Edge (10, 2507) - Weight: 0.5472
    Edge (10, 2508) - Weight: 0.5692
    Edge (10, 2510) - Weight: 0.5710
    Edge (10, 2524) - Weight: 0.6027
    Edge (10, 2526) - Weight: 0.5191
    Edge (10, 2527) - Weight: 0.6197
    Edge (10, 2533) - Weight: 0.5435
    Edge (10, 2537) - Weight: 0.5059
    Edge (10, 3729) - Weight: 0.5566
    Edge (13, 16) - Weight: 0.5820
    Edge (13, 27) - Weight: 0.6093
    Edge (13, 35) - Weight: 0.5063
    Edge (13, 49) - Weight: 0.5831
    Edge (13, 2073) - Weight: 0.5069
    Edge (13, 2507) - Weight: 0.5327
    Edge (13, 2508) - Weight: 0.5295
    Edge (13, 2510) - Weight: 0.6055
    Edge (13, 2524) - Weight: 0.5599
    Edge (13, 2526) - Weight: 0.6181
    Edge (13, 2527) - Weight: 0.5321
    Edge (13, 2533) - Weight: 0.5323
    Edge (13, 2537) - Weight: 0.6228
    Edge (14, 30) - Weight: 0.6725
    Edge (14, 47) - Weight: 0.7041
    Edge (15, 4553) - Weight: 0.5220
    Edge (16, 27) - Weight: 0.6055
    Edge (16, 49) - Weight: 0.5468
    Edge (16, 2510) - Weight: 0.5477
    Edge (16, 2524) - Weight: 0.5025
    Edge (16, 2526) - Weight: 0.5276
    Edge (16, 2537) - Weight: 0.5338
    Edge (17, 2092) - Weight: 0.5458
    Edge (18, 19) - Weight: 0.6412
    Edge (18, 23) - Weight: 0.7034
    Edge (18, 24) - Weight: 0.6561
    Edge (18, 40) - Weight: 0.5320
    Edge (18, 2515) - Weight: 0.5048
    Edge (18, 2516) - Weight: 0.5377
    Edge (19, 23) - Weight: 0.7906
    Edge (19, 24) - Weight: 0.9440
    Edge (19, 2515) - Weight: 0.5544
    Edge (19, 2516) - Weight: 0.5972
    Edge (21, 22) - Weight: 0.9381
    Edge (21, 38) - Weight: 0.6512
    Edge (21, 2518) - Weight: 0.6474
    Edge (21, 2519) - Weight: 0.6216
    Edge (21, 2520) - Weight: 0.5454
    Edge (21, 2521) - Weight: 0.6506
    Edge (22, 38) - Weight: 0.6648
    Edge (22, 2518) - Weight: 0.6527
    Edge (22, 2519) - Weight: 0.6277
    Edge (22, 2520) - Weight: 0.5443
    Edge (22, 2521) - Weight: 0.6498
    Edge (23, 24) - Weight: 0.8176
    Edge (23, 2515) - Weight: 0.5882
    Edge (23, 2516) - Weight: 0.6541
    Edge (24, 2515) - Weight: 0.5541
    Edge (24, 2516) - Weight: 0.6308
    Edge (25, 26) - Weight: 0.6840
    Edge (25, 2000) - Weight: 0.5133
    Edge (25, 2042) - Weight: 0.5243
    Edge (25, 4530) - Weight: 0.5873
    Edge (25, 4531) - Weight: 0.5824
    Edge (25, 4535) - Weight: 0.5270
    Edge (25, 4536) - Weight: 0.5608
    Edge (26, 31) - Weight: 0.5013
    Edge (26, 32) - Weight: 0.5090
    Edge (26, 2335) - Weight: 0.5016
    Edge (27, 49) - Weight: 0.5022
    Edge (27, 2510) - Weight: 0.6314
    Edge (27, 2524) - Weight: 0.5438
    Edge (27, 2526) - Weight: 0.5320
    Edge (27, 2527) - Weight: 0.5111
    Edge (27, 2537) - Weight: 0.6479
    Edge (28, 33) - Weight: 0.5274
    

### Graph Connectivity
The total number of connected components in the graph is printed, along with the percentage of nodes that have at least one edge, indicating how well-connected the graph is.


```python
num_connected_components = nx.number_connected_components(G)
print(f"Total number of connected components: {num_connected_components}")

nodes_with_edges = len([node for node in G.nodes if G.degree(node) > 0])
percent_nodes_with_edges = (nodes_with_edges / G.number_of_nodes()) * 100
print(f"Percentage of nodes with at least one edge: {percent_nodes_with_edges:.2f}%")
```

    Total number of connected components: 880
    Percentage of nodes with at least one edge: 88.32%
    

### Gephi Exportation

This code is just to help export our graph to Gephi for deeper analysis.


```python
import scipy.sparse 
for node, attr in G.nodes(data=True):
    if 'feature' in attr and isinstance(attr['feature'], (scipy.sparse.spmatrix, np.ndarray)):
        attr['feature'] = attr['feature'].toarray().tolist() if isinstance(attr['feature'], scipy.sparse.spmatrix) else attr['feature'].tolist()


for u, v, attr in G.edges(data=True):
    if 'weight' in attr and isinstance(attr['weight'], (scipy.sparse.spmatrix, np.ndarray)):
        attr['weight'] = attr['weight'].tolist()



```


```python
import networkx as nx

G_original = G.copy()

G_modified = G_original.copy()

for node in G_modified.nodes(data=True):
    if 'feature' in node[1]:  
        del node[1]['feature']

for u, v, attr in G_modified.edges(data=True):
    if 'weight' in attr:  
        del attr['weight']

```


```python
output_file = "graph_export.gexf"
nx.write_gexf(G_modified, output_file)

print(f"Modified graph exported to {output_file}")
```

    Modified graph exported to graph_export.gexf
    

### Converting Graph and Preparing Data
The NetworkX graph is converted to a PyTorch Geometric Data object. Node features (data.x) are assigned using TF-IDF vectors, and labels (data.y) are set as the encoded author labels. Masks are created to distinguish training and test nodes, where the training set includes the first portion and the test set includes the rest.




```python

data = from_networkx(G)


data.x = torch.tensor(features.todense(), dtype=torch.float)
data.y = torch.tensor(labels, dtype=torch.long)


data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.train_mask[:len(train_texts)] = 1

data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.test_mask[len(train_texts):] = 1

```

### Graph Convolutional Network (GCN) Model
A two-layer GCN model is defined, where the first layer (conv1) applies a graph convolution to the node features, followed by ReLU activation and dropout. The second layer (conv2) performs another convolution, and the output is passed through a softmax for classification. The model is initialized with input dimensions based on TF-IDF features, hidden dimensions of size 64, and output dimensions equal to the number of authors.


```python

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.log_softmax(x, dim=1)


input_dim = data.num_features
hidden_dim = 64
output_dim = len(label_encoder.classes_)


model = GCN(input_dim, hidden_dim, output_dim)

```

### Model Training, Testing, and Evaluation
This code defines the training and testing loop for the GCN model. The train function performs a forward pass, computes the loss, and updates the model weights. The test function evaluates the model by calculating accuracy on the test set and displays predictions.

Additionally, two functions are provided for visualization:

Confusion Matrix: A heatmap visualizing the predicted vs. true author labels.
Per-Class Accuracy: A bar chart showing the accuracy for each author individually.

The model is trained for 200 epochs, with testing and accuracy results printed every 10 epochs.

*There is extensive output as the visualizations are printed each 10 epochs. We used this for studying how the model learned*


```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(data, pred):
    cm = confusion_matrix(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test():
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())
    
    print("\nFull Test Set Predictions:")
    test_indices = data.test_mask.nonzero(as_tuple=True)[0]
    for idx in test_indices:
        true_label = data.y[idx].item()
        predicted_label = pred[idx].item()
        print(f"Index: {idx}, True Label: {true_label}, Predicted Label: {predicted_label}")

    per_class_accuracy(data, pred)
    plot_confusion_matrix(data, pred)
    
    return acc


epochs = 200
for epoch in range(epochs):
    loss = train()
    if epoch % 10 == 0:
        test_acc = test()  
        print(f'Epoch: {epoch}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f}')


test_acc = test() 
print(f'Final Test Accuracy: {test_acc:.4f}')

def per_class_accuracy(data, pred):
    class_correct = np.zeros(len(label_encoder.classes_))
    class_total = np.zeros(len(label_encoder.classes_))
    
    for i in range(len(data.y[data.test_mask])):
        label = data.y[data.test_mask][i].item()
        class_total[label] += 1
        if pred[data.test_mask][i].item() == label:
            class_correct[label] += 1

    class_accuracy = class_correct / class_total
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(label_encoder.classes_)), class_accuracy, tick_label=label_encoder.classes_)
    plt.xlabel('Authors')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.xticks(rotation=90)
    plt.show()

    return class_accuracy




```

    
   
    Index: 4950, True Label: 49, Predicted Label: 43
    Index: 4951, True Label: 49, Predicted Label: 11
    Index: 4952, True Label: 49, Predicted Label: 11
    Index: 4953, True Label: 49, Predicted Label: 45
    Index: 4954, True Label: 49, Predicted Label: 27
    Index: 4955, True Label: 49, Predicted Label: 14
    Index: 4956, True Label: 49, Predicted Label: 43
    Index: 4957, True Label: 49, Predicted Label: 11
    Index: 4958, True Label: 49, Predicted Label: 43
    Index: 4959, True Label: 49, Predicted Label: 49
    Index: 4960, True Label: 49, Predicted Label: 3
    Index: 4961, True Label: 49, Predicted Label: 49
    Index: 4962, True Label: 49, Predicted Label: 49
    Index: 4963, True Label: 49, Predicted Label: 34
    Index: 4964, True Label: 49, Predicted Label: 43
    Index: 4965, True Label: 49, Predicted Label: 34
    Index: 4966, True Label: 49, Predicted Label: 49
    Index: 4967, True Label: 49, Predicted Label: 27
    Index: 4968, True Label: 49, Predicted Label: 34
    Index: 4969, True Label: 49, Predicted Label: 11
    Index: 4970, True Label: 49, Predicted Label: 37
    Index: 4971, True Label: 49, Predicted Label: 34
    Index: 4972, True Label: 49, Predicted Label: 3
    Index: 4973, True Label: 49, Predicted Label: 49
    Index: 4974, True Label: 49, Predicted Label: 34
    Index: 4975, True Label: 49, Predicted Label: 34
    Index: 4976, True Label: 49, Predicted Label: 34
    Index: 4977, True Label: 49, Predicted Label: 3
    Index: 4978, True Label: 49, Predicted Label: 49
    Index: 4979, True Label: 49, Predicted Label: 45
    Index: 4980, True Label: 49, Predicted Label: 37
    Index: 4981, True Label: 49, Predicted Label: 49
    Index: 4982, True Label: 49, Predicted Label: 49
    Index: 4983, True Label: 49, Predicted Label: 3
    Index: 4984, True Label: 49, Predicted Label: 31
    Index: 4985, True Label: 49, Predicted Label: 49
    Index: 4986, True Label: 49, Predicted Label: 3
    Index: 4987, True Label: 49, Predicted Label: 45
    Index: 4988, True Label: 49, Predicted Label: 49
    Index: 4989, True Label: 49, Predicted Label: 49
    Index: 4990, True Label: 49, Predicted Label: 49
    Index: 4991, True Label: 49, Predicted Label: 30
    Index: 4992, True Label: 49, Predicted Label: 49
    Index: 4993, True Label: 49, Predicted Label: 49
    Index: 4994, True Label: 49, Predicted Label: 3
    Index: 4995, True Label: 49, Predicted Label: 49
    Index: 4996, True Label: 49, Predicted Label: 49
    Index: 4997, True Label: 49, Predicted Label: 49
    Index: 4998, True Label: 49, Predicted Label: 11
    Index: 4999, True Label: 49, Predicted Label: 3
    


    
![png](/assets/reutersCorpus_files/reutersCorpus_22_61.png)
    



    
![png](/assets/reutersCorpus_files/reutersCorpus_22_62.png)
    


    Final Test Accuracy: 0.6260
    

### Print Top-5 Accuracy


```python
def top_k_accuracy(data, k=5):
    model.eval()
    out = model(data)
    topk_pred = out.topk(k, dim=1).indices
    correct = 0
    
    for i in range(len(data.y[data.test_mask])):
        if data.y[data.test_mask][i].item() in topk_pred[data.test_mask][i].cpu().numpy():
            correct += 1
    
    acc = correct / int(data.test_mask.sum())
    print(f'Top-{k} Accuracy: {acc:.4f}')
    return acc


top_k_accuracy(data, k=5)

```

    Top-5 Accuracy: 0.9420
    




    0.942



### Attempting GCN with GAT Layers
This code defines an improved GCN model with the option to use Graph Attention Network (GAT) layers. The model has three layers, either GCN or GAT depending on the use_gat flag. If GAT is used, each layer has 8 attention heads for better feature aggregation from neighboring nodes. ReLU activation and dropout are applied between layers. The model's output is a softmax over the predicted classes (authors).

The model is initialized with two hidden layers of 64 and 32 units, and the output corresponds to the number of authors in the dataset.


```python
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class ImprovedGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, use_gat=False):
        super(ImprovedGCN, self).__init__()
        self.use_gat = use_gat
        
        if use_gat:
            self.conv1 = GATConv(input_dim, hidden_dim1, heads=8, concat=True)
            self.conv2 = GATConv(hidden_dim1 * 8, hidden_dim2, heads=8, concat=True)
            self.conv3 = GATConv(hidden_dim2 * 8, output_dim, heads=8, concat=False)
        else:
            self.conv1 = GCNConv(input_dim, hidden_dim1)
            self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
            self.conv3 = GCNConv(hidden_dim2, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv3(x, edge_index)
        
        return F.log_softmax(x, dim=1)


input_dim = data.num_features
hidden_dim1 = 64
hidden_dim2 = 32
output_dim = len(label_encoder.classes_)

## This is attempting with GAT layers
model = ImprovedGCN(input_dim, hidden_dim1, hidden_dim2, output_dim, use_gat=True)

```

### Training and Testing Loop
This code defines the training and evaluation process for the GAT model. The Adam optimizer is used with a learning rate of 0.01 and weight decay of 5e-4. The Cross-Entropy Loss function is applied to compute the loss during training. The train function updates the model parameters, while the test function evaluates accuracy on the test set. The model is trained for 200 epochs, with accuracy printed every 10 epochs. We ended with lower accuracy than the regular GCN so abandoned this tactic.


```python

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test():
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())
    return acc


epochs = 200
for epoch in range(epochs):
    loss = train()
    if epoch % 10 == 0:
        test_acc = test()
        print(f'Epoch: {epoch}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f}')


test_acc = test()
print(f'Final Test Accuracy: {test_acc:.4f}')

```

    Epoch: 0, Loss: 3.9118, Test Accuracy: 0.1500
    Epoch: 10, Loss: 1.6232, Test Accuracy: 0.3560
    Epoch: 20, Loss: 0.7443, Test Accuracy: 0.5528
    Epoch: 30, Loss: 0.6076, Test Accuracy: 0.5636
    Epoch: 40, Loss: 0.5388, Test Accuracy: 0.5668
    Epoch: 50, Loss: 0.4903, Test Accuracy: 0.5744
    Epoch: 60, Loss: 0.4924, Test Accuracy: 0.5736
    Epoch: 70, Loss: 0.4749, Test Accuracy: 0.5788
    Epoch: 80, Loss: 0.4522, Test Accuracy: 0.5644
    Epoch: 90, Loss: 0.4578, Test Accuracy: 0.5732
    Epoch: 100, Loss: 0.4606, Test Accuracy: 0.5788
    Epoch: 110, Loss: 0.4587, Test Accuracy: 0.5752
    Epoch: 120, Loss: 0.4379, Test Accuracy: 0.5652
    Epoch: 130, Loss: 0.4449, Test Accuracy: 0.5668
    Epoch: 140, Loss: 0.4400, Test Accuracy: 0.5836
    Epoch: 150, Loss: 0.4559, Test Accuracy: 0.5684
    Epoch: 160, Loss: 0.4329, Test Accuracy: 0.5668
    Epoch: 170, Loss: 0.4357, Test Accuracy: 0.5756
    Epoch: 180, Loss: 0.4409, Test Accuracy: 0.5656
    Epoch: 190, Loss: 0.4383, Test Accuracy: 0.5836
    Final Test Accuracy: 0.5632
    

### Trying Different Thresholds
Same data loadings process, except we downsampled to only 10 documents per author in each of the test and train folders to accelerate the process.


```python
import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import numpy as np
import random
import time

def load_data(root_dir, sample_size=10):
    texts = []
    labels = []
    author_folders = os.listdir(root_dir)
    
    for author in author_folders:
        author_dir = os.path.join(root_dir, author)
        files = os.listdir(author_dir)
        sampled_files = random.sample(files, sample_size)
        
        for filename in sampled_files:
            with open(os.path.join(author_dir, filename), 'r', encoding='latin-1') as file:
                texts.append(file.read())
                labels.append(author)
    
    return texts, labels


train_texts, train_labels = load_data('data/ReutersC50/C50train', sample_size=10)
test_texts, test_labels = load_data('data/ReutersC50/C50test', sample_size=10)


all_texts = train_texts + test_texts
all_labels = train_labels + test_labels


label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(all_labels)


vectorizer = TfidfVectorizer(max_features=5000)
features = vectorizer.fit_transform(all_texts)

```

Same graph creation, just with less data and altered threshold


```python


G_new_thresh = nx.Graph()


print("Adding nodes to the graph...")
start_time = time.time()

for i in range(features.shape[0]):
    G_new_thresh.add_node(i, feature=features[i], label=labels[i])

end_time = time.time()
print(f"Nodes added: {features.shape[0]} nodes in {end_time - start_time:.2f} seconds.")

print("Adding edges to the graph based on cosine similarity...")
start_time = time.time()

threshold = 0.7  ## ADJUSTED THRESHOLD
total_comparisons = (features.shape[0] * (features.shape[0] - 1)) // 2
comparison_count = 0

for i in range(features.shape[0]):
    for j in range(i + 1, features.shape[0]):
        similarity = cosine_similarity(features[i], features[j])[0][0]
        if similarity > threshold:
            G_new_thresh.add_edge(i, j, weight=similarity)
        comparison_count += 1
        
        
        if comparison_count % 100000 == 0:
            elapsed_time = time.time() - start_time
            print(f"Processed {comparison_count}/{total_comparisons} comparisons "
                  f"({(comparison_count / total_comparisons) * 100:.2f}%) "
                  f"in {elapsed_time:.2f} seconds.")

end_time = time.time()
print(f"Edges added in {end_time - start_time:.2f} seconds.")

```

    Adding nodes to the graph...
    Nodes added: 1000 nodes in 0.13 seconds.
    Adding edges to the graph based on cosine similarity...
    Processed 100000/499500 comparisons (20.02%) in 72.34 seconds.
    Processed 200000/499500 comparisons (40.04%) in 143.47 seconds.
    Processed 300000/499500 comparisons (60.06%) in 218.98 seconds.
    Processed 400000/499500 comparisons (80.08%) in 296.64 seconds.
    Edges added in 376.69 seconds.
    

### Threshold Testing
This code generates multiple graphs based on different cosine similarity thresholds (0.7, 0.8, and 0.9).



```python



thresholds = [.7, 0.8, .9]


graphs = {}


for threshold in thresholds:
    G1 = nx.Graph()
    
    print(f"\nCreating graph with threshold: {threshold}")
    
  
    print("Adding nodes to the graph...")
    start_time = time.time()
    
    for i in range(features.shape[0]):
        G1.add_node(i, feature=features[i], label=labels[i])
    
    end_time = time.time()
    print(f"Nodes added: {features.shape[0]} nodes in {end_time - start_time:.2f} seconds.")
    
  
    print(f"Adding edges to the graph based on cosine similarity with threshold {threshold}...")
    start_time = time.time()
    
    total_comparisons = (features.shape[0] * (features.shape[0] - 1)) // 2
    comparison_count = 0
    
    for i in range(features.shape[0]):
        for j in range(i + 1, features.shape[0]):
            similarity = cosine_similarity(features[i], features[j])[0][0]
            if similarity > threshold:
                G1.add_edge(i, j, weight=similarity)
            comparison_count += 1
            
      
            if comparison_count % 100000 == 0:
                elapsed_time = time.time() - start_time
                print(f"Processed {comparison_count}/{total_comparisons} comparisons "
                      f"({(comparison_count / total_comparisons) * 100:.2f}%) "
                      f"in {elapsed_time:.2f} seconds.")
    
    end_time = time.time()
    print(f"Edges added in {end_time - start_time:.2f} seconds.")
    
    
    graphs[threshold] = G1


for threshold, graph in graphs.items():
    print(f"\nGraph for threshold {threshold} has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

```

    
    Creating graph with threshold: 0.7
    Adding nodes to the graph...
    Nodes added: 1000 nodes in 0.07 seconds.
    Adding edges to the graph based on cosine similarity with threshold 0.7...
    Processed 100000/499500 comparisons (20.02%) in 71.64 seconds.
    Processed 200000/499500 comparisons (40.04%) in 142.79 seconds.
    Processed 300000/499500 comparisons (60.06%) in 214.60 seconds.
    Processed 400000/499500 comparisons (80.08%) in 282.95 seconds.
    Edges added in 350.91 seconds.
    
    Creating graph with threshold: 0.8
    Adding nodes to the graph...
    Nodes added: 1000 nodes in 0.06 seconds.
    Adding edges to the graph based on cosine similarity with threshold 0.8...
    Processed 100000/499500 comparisons (20.02%) in 70.07 seconds.
    Processed 200000/499500 comparisons (40.04%) in 141.40 seconds.
    Processed 300000/499500 comparisons (60.06%) in 210.41 seconds.
    Processed 400000/499500 comparisons (80.08%) in 279.72 seconds.
    Edges added in 347.25 seconds.
    
    Creating graph with threshold: 0.9
    Adding nodes to the graph...
    Nodes added: 1000 nodes in 0.06 seconds.
    Adding edges to the graph based on cosine similarity with threshold 0.9...
    Processed 100000/499500 comparisons (20.02%) in 69.49 seconds.
    Processed 200000/499500 comparisons (40.04%) in 138.41 seconds.
    Processed 300000/499500 comparisons (60.06%) in 207.31 seconds.
    Processed 400000/499500 comparisons (80.08%) in 276.39 seconds.
    Edges added in 345.76 seconds.
    
    Graph for threshold 0.7 has 1000 nodes and 201 edges.
    
    Graph for threshold 0.8 has 1000 nodes and 74 edges.
    
    Graph for threshold 0.9 has 1000 nodes and 48 edges.
    

### Training and Evaluating with Multiple Thresholds
This code trains a Graph Convolutional Network (GCN) for each graph generated with different cosine similarity thresholds (0.7, 0.8, 0.9). The code identifies and prints the threshold that yields the highest test accuracy across all graphs.



```python
def train_and_evaluate(threshold, graph):
    print(f"\nTraining model for graph with threshold: {threshold}")
    
   
    data = from_networkx(graph)
    data.x = torch.tensor(features.todense(), dtype=torch.float)
    data.y = torch.tensor(labels, dtype=torch.long)

    
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[:len(train_texts)] = 1
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask[len(train_texts):] = 1

 
    class GCN(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, output_dim)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = self.conv1(x, edge_index)
            x = torch.relu(x)
            x = torch.nn.functional.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            return torch.log_softmax(x, dim=1)

  
    input_dim = data.num_features
    hidden_dim = 64
    output_dim = len(label_encoder.classes_)

   
    model = GCN(input_dim, hidden_dim, output_dim)

  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

  
    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

   
    def test():
        model.eval()
        _, pred = model(data).max(dim=1)
        correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        acc = correct / int(data.test_mask.sum())
        return acc

   
    epochs = 200 
    best_acc = 0
    for epoch in range(epochs):
        loss = train()
        if epoch % 10 == 0:
            test_acc = test()
            print(f'Epoch: {epoch}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f}')
            best_acc = max(best_acc, test_acc)
    
    return best_acc

best_threshold = None
best_accuracy = 0

for threshold, graph in graphs.items():
    accuracy = train_and_evaluate(threshold, graph)
    print(f"Threshold {threshold} achieved an accuracy of {accuracy:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

print(f"\nBest Threshold: {best_threshold} with Test Accuracy: {best_accuracy:.4f}")
```

    
    Training model for graph with threshold: 0.7
    Epoch: 0, Loss: 3.9130, Test Accuracy: 0.1680
    Epoch: 10, Loss: 2.4958, Test Accuracy: 0.5180
    Epoch: 20, Loss: 0.9187, Test Accuracy: 0.6220
    Epoch: 30, Loss: 0.3575, Test Accuracy: 0.6000
    Epoch: 40, Loss: 0.2864, Test Accuracy: 0.5960
    Epoch: 50, Loss: 0.3114, Test Accuracy: 0.5960
    Epoch: 60, Loss: 0.2647, Test Accuracy: 0.6040
    Epoch: 70, Loss: 0.2368, Test Accuracy: 0.6000
    Epoch: 80, Loss: 0.2355, Test Accuracy: 0.6020
    Epoch: 90, Loss: 0.2188, Test Accuracy: 0.5960
    Epoch: 100, Loss: 0.2389, Test Accuracy: 0.6000
    Epoch: 110, Loss: 0.2271, Test Accuracy: 0.6000
    Epoch: 120, Loss: 0.2050, Test Accuracy: 0.5940
    Epoch: 130, Loss: 0.1966, Test Accuracy: 0.6040
    Epoch: 140, Loss: 0.2052, Test Accuracy: 0.5920
    Epoch: 150, Loss: 0.1990, Test Accuracy: 0.5980
    Epoch: 160, Loss: 0.1735, Test Accuracy: 0.5900
    Epoch: 170, Loss: 0.1815, Test Accuracy: 0.5940
    Epoch: 180, Loss: 0.1874, Test Accuracy: 0.6060
    Epoch: 190, Loss: 0.2092, Test Accuracy: 0.5960
    Threshold 0.7 achieved an accuracy of 0.6220
    
    Training model for graph with threshold: 0.8
    Epoch: 0, Loss: 3.9115, Test Accuracy: 0.1420
    Epoch: 10, Loss: 2.4670, Test Accuracy: 0.5100
    Epoch: 20, Loss: 0.8872, Test Accuracy: 0.5940
    Epoch: 30, Loss: 0.3577, Test Accuracy: 0.6080
    Epoch: 40, Loss: 0.2773, Test Accuracy: 0.5880
    Epoch: 50, Loss: 0.2878, Test Accuracy: 0.5860
    Epoch: 60, Loss: 0.2824, Test Accuracy: 0.5840
    Epoch: 70, Loss: 0.2379, Test Accuracy: 0.5840
    Epoch: 80, Loss: 0.2254, Test Accuracy: 0.5960
    Epoch: 90, Loss: 0.1976, Test Accuracy: 0.5840
    Epoch: 100, Loss: 0.2139, Test Accuracy: 0.5920
    Epoch: 110, Loss: 0.1849, Test Accuracy: 0.5860
    Epoch: 120, Loss: 0.1936, Test Accuracy: 0.5840
    Epoch: 130, Loss: 0.1959, Test Accuracy: 0.5880
    Epoch: 140, Loss: 0.1789, Test Accuracy: 0.5920
    Epoch: 150, Loss: 0.1702, Test Accuracy: 0.5800
    Epoch: 160, Loss: 0.1608, Test Accuracy: 0.5840
    Epoch: 170, Loss: 0.1774, Test Accuracy: 0.5940
    Epoch: 180, Loss: 0.1510, Test Accuracy: 0.5980
    Epoch: 190, Loss: 0.1639, Test Accuracy: 0.5860
    Threshold 0.8 achieved an accuracy of 0.6080
    
    Training model for graph with threshold: 0.9
    Epoch: 0, Loss: 3.9121, Test Accuracy: 0.1420
    Epoch: 10, Loss: 2.4579, Test Accuracy: 0.4960
    Epoch: 20, Loss: 0.8850, Test Accuracy: 0.5800
    Epoch: 30, Loss: 0.3721, Test Accuracy: 0.5800
    Epoch: 40, Loss: 0.2537, Test Accuracy: 0.5800
    Epoch: 50, Loss: 0.2971, Test Accuracy: 0.5740
    Epoch: 60, Loss: 0.2641, Test Accuracy: 0.5760
    Epoch: 70, Loss: 0.2261, Test Accuracy: 0.5880
    Epoch: 80, Loss: 0.2265, Test Accuracy: 0.5760
    Epoch: 90, Loss: 0.2068, Test Accuracy: 0.5840
    Epoch: 100, Loss: 0.2004, Test Accuracy: 0.5820
    Epoch: 110, Loss: 0.1922, Test Accuracy: 0.5780
    Epoch: 120, Loss: 0.1793, Test Accuracy: 0.5740
    Epoch: 130, Loss: 0.1811, Test Accuracy: 0.5740
    Epoch: 140, Loss: 0.1807, Test Accuracy: 0.5760
    Epoch: 150, Loss: 0.1641, Test Accuracy: 0.5860
    Epoch: 160, Loss: 0.1887, Test Accuracy: 0.6060
    Epoch: 170, Loss: 0.1743, Test Accuracy: 0.5780
    Epoch: 180, Loss: 0.1846, Test Accuracy: 0.5800
    Epoch: 190, Loss: 0.1786, Test Accuracy: 0.5940
    Threshold 0.9 achieved an accuracy of 0.6060
    
    Best Threshold: 0.7 with Test Accuracy: 0.6220
    

Now trying threshold .7 on full set.


```python
import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import numpy as np

def load_data(root_dir):
    texts = []
    labels = []
    author_folders = os.listdir(root_dir)
    
    for author in author_folders:
        author_dir = os.path.join(root_dir, author)
        for filename in os.listdir(author_dir):
            with open(os.path.join(author_dir, filename), 'r', encoding='latin-1') as file:
                texts.append(file.read())
                labels.append(author)
    
    return texts, labels

train_texts, train_labels = load_data('data/ReutersC50/C50train')
test_texts, test_labels = load_data('data/ReutersC50/C50test')


all_texts = train_texts + test_texts
all_labels = train_labels + test_labels

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(all_labels)

vectorizer = TfidfVectorizer(max_features=5000)
features = vectorizer.fit_transform(all_texts)
```


```python
import time


G_7 = nx.Graph()


print("Adding nodes to the graph...")
start_time = time.time()

for i in range(features.shape[0]):
    G_7.add_node(i, feature=features[i], label=labels[i])

end_time = time.time()
print(f"Nodes added: {features.shape[0]} nodes in {end_time - start_time:.2f} seconds.")

## We want to add edges based on cosine similarity
print("Adding edges to the graph based on cosine similarity...")
start_time = time.time()

threshold = 0.7 
total_comparisons = (features.shape[0] * (features.shape[0] - 1)) // 2
comparison_count = 0

for i in range(features.shape[0]):
    for j in range(i + 1, features.shape[0]):
        similarity = cosine_similarity(features[i], features[j])[0][0]
        if similarity > threshold:
            G_7.add_edge(i, j, weight=similarity)
        comparison_count += 1
        
        ## Just wanted progress updates
        if comparison_count % 100000 == 0:
            elapsed_time = time.time() - start_time
            print(f"Processed {comparison_count}/{total_comparisons} comparisons "
                  f"({(comparison_count / total_comparisons) * 100:.2f}%) "
                  f"in {elapsed_time:.2f} seconds.")

end_time = time.time()
print(f"Edges added in {end_time - start_time:.2f} seconds.")

```

    Adding nodes to the graph...
    Nodes added: 5000 nodes in 0.30 seconds.
    Adding edges to the graph based on cosine similarity...
    Processed 100000/12497500 comparisons (0.80%) in 69.31 seconds.
    Processed 200000/12497500 comparisons (1.60%) in 137.00 seconds.
    Processed 300000/12497500 comparisons (2.40%) in 205.22 seconds.
    Processed 400000/12497500 comparisons (3.20%) in 272.74 seconds.
    Processed 500000/12497500 comparisons (4.00%) in 340.64 seconds.
    Processed 600000/12497500 comparisons (4.80%) in 409.15 seconds.
    Processed 700000/12497500 comparisons (5.60%) in 479.09 seconds.
    Processed 800000/12497500 comparisons (6.40%) in 546.96 seconds.
    Processed 900000/12497500 comparisons (7.20%) in 614.90 seconds.
    Processed 1000000/12497500 comparisons (8.00%) in 682.51 seconds.
    Processed 1100000/12497500 comparisons (8.80%) in 750.93 seconds.
    Processed 1200000/12497500 comparisons (9.60%) in 818.62 seconds.
    Processed 1300000/12497500 comparisons (10.40%) in 887.75 seconds.
    Processed 1400000/12497500 comparisons (11.20%) in 955.73 seconds.
    Processed 1500000/12497500 comparisons (12.00%) in 1023.76 seconds.
    Processed 1600000/12497500 comparisons (12.80%) in 1091.34 seconds.
    Processed 1700000/12497500 comparisons (13.60%) in 1158.98 seconds.
    Processed 1800000/12497500 comparisons (14.40%) in 1227.50 seconds.
    Processed 1900000/12497500 comparisons (15.20%) in 1296.12 seconds.
    Processed 2000000/12497500 comparisons (16.00%) in 1365.19 seconds.
    Processed 2100000/12497500 comparisons (16.80%) in 1433.79 seconds.
    Processed 2200000/12497500 comparisons (17.60%) in 1502.18 seconds.
    Processed 2300000/12497500 comparisons (18.40%) in 1570.78 seconds.
    Processed 2400000/12497500 comparisons (19.20%) in 1639.12 seconds.
    Processed 2500000/12497500 comparisons (20.00%) in 1707.56 seconds.
    Processed 2600000/12497500 comparisons (20.80%) in 1775.86 seconds.
    Processed 2700000/12497500 comparisons (21.60%) in 1844.68 seconds.
    Processed 2800000/12497500 comparisons (22.40%) in 1913.02 seconds.
    Processed 2900000/12497500 comparisons (23.20%) in 1981.64 seconds.
    Processed 3000000/12497500 comparisons (24.00%) in 2050.12 seconds.
    Processed 3100000/12497500 comparisons (24.80%) in 2118.32 seconds.
    Processed 3200000/12497500 comparisons (25.61%) in 2186.55 seconds.
    Processed 3300000/12497500 comparisons (26.41%) in 2254.73 seconds.
    Processed 3400000/12497500 comparisons (27.21%) in 2323.80 seconds.
    Processed 3500000/12497500 comparisons (28.01%) in 2391.91 seconds.
    Processed 3600000/12497500 comparisons (28.81%) in 2460.77 seconds.
    Processed 3700000/12497500 comparisons (29.61%) in 2528.93 seconds.
    Processed 3800000/12497500 comparisons (30.41%) in 2596.59 seconds.
    Processed 3900000/12497500 comparisons (31.21%) in 2664.59 seconds.
    Processed 4000000/12497500 comparisons (32.01%) in 2732.34 seconds.
    Processed 4100000/12497500 comparisons (32.81%) in 2800.16 seconds.
    Processed 4200000/12497500 comparisons (33.61%) in 2868.28 seconds.
    Processed 4300000/12497500 comparisons (34.41%) in 2936.38 seconds.
    Processed 4400000/12497500 comparisons (35.21%) in 3005.04 seconds.
    Processed 4500000/12497500 comparisons (36.01%) in 3073.61 seconds.
    Processed 4600000/12497500 comparisons (36.81%) in 3141.98 seconds.
    Processed 4700000/12497500 comparisons (37.61%) in 3211.18 seconds.
    Processed 4800000/12497500 comparisons (38.41%) in 3279.77 seconds.
    Processed 4900000/12497500 comparisons (39.21%) in 3348.35 seconds.
    Processed 5000000/12497500 comparisons (40.01%) in 3416.58 seconds.
    Processed 5100000/12497500 comparisons (40.81%) in 3484.74 seconds.
    Processed 5200000/12497500 comparisons (41.61%) in 3552.94 seconds.
    Processed 5300000/12497500 comparisons (42.41%) in 3621.29 seconds.
    Processed 5400000/12497500 comparisons (43.21%) in 3689.40 seconds.
    Processed 5500000/12497500 comparisons (44.01%) in 3757.20 seconds.
    Processed 5600000/12497500 comparisons (44.81%) in 3825.41 seconds.
    Processed 5700000/12497500 comparisons (45.61%) in 3893.66 seconds.
    Processed 5800000/12497500 comparisons (46.41%) in 3962.30 seconds.
    Processed 5900000/12497500 comparisons (47.21%) in 4030.90 seconds.
    Processed 6000000/12497500 comparisons (48.01%) in 4099.84 seconds.
    Processed 6100000/12497500 comparisons (48.81%) in 4168.43 seconds.
    Processed 6200000/12497500 comparisons (49.61%) in 4236.97 seconds.
    Processed 6300000/12497500 comparisons (50.41%) in 4305.35 seconds.
    Processed 6400000/12497500 comparisons (51.21%) in 4374.50 seconds.
    Processed 6500000/12497500 comparisons (52.01%) in 4443.08 seconds.
    Processed 6600000/12497500 comparisons (52.81%) in 4512.14 seconds.
    Processed 6700000/12497500 comparisons (53.61%) in 4580.08 seconds.
    Processed 6800000/12497500 comparisons (54.41%) in 4648.13 seconds.
    Processed 6900000/12497500 comparisons (55.21%) in 4716.41 seconds.
    Processed 7000000/12497500 comparisons (56.01%) in 4784.36 seconds.
    Processed 7100000/12497500 comparisons (56.81%) in 4851.86 seconds.
    Processed 7200000/12497500 comparisons (57.61%) in 4919.52 seconds.
    Processed 7300000/12497500 comparisons (58.41%) in 4987.75 seconds.
    Processed 7400000/12497500 comparisons (59.21%) in 5055.47 seconds.
    Processed 7500000/12497500 comparisons (60.01%) in 5123.21 seconds.
    Processed 7600000/12497500 comparisons (60.81%) in 5191.61 seconds.
    Processed 7700000/12497500 comparisons (61.61%) in 5259.95 seconds.
    Processed 7800000/12497500 comparisons (62.41%) in 5328.01 seconds.
    Processed 7900000/12497500 comparisons (63.21%) in 5396.64 seconds.
    Processed 8000000/12497500 comparisons (64.01%) in 5465.39 seconds.
    Processed 8100000/12497500 comparisons (64.81%) in 5533.73 seconds.
    Processed 8200000/12497500 comparisons (65.61%) in 5602.55 seconds.
    Processed 8300000/12497500 comparisons (66.41%) in 5671.23 seconds.
    Processed 8400000/12497500 comparisons (67.21%) in 5739.50 seconds.
    Processed 8500000/12497500 comparisons (68.01%) in 5807.67 seconds.
    Processed 8600000/12497500 comparisons (68.81%) in 5876.24 seconds.
    Processed 8700000/12497500 comparisons (69.61%) in 5944.44 seconds.
    Processed 8800000/12497500 comparisons (70.41%) in 6012.52 seconds.
    Processed 8900000/12497500 comparisons (71.21%) in 6080.52 seconds.
    Processed 9000000/12497500 comparisons (72.01%) in 6148.80 seconds.
    Processed 9100000/12497500 comparisons (72.81%) in 6216.63 seconds.
    Processed 9200000/12497500 comparisons (73.61%) in 6284.47 seconds.
    Processed 9300000/12497500 comparisons (74.41%) in 6352.36 seconds.
    Processed 9400000/12497500 comparisons (75.22%) in 6420.32 seconds.
    Processed 9500000/12497500 comparisons (76.02%) in 6488.32 seconds.
    Processed 9600000/12497500 comparisons (76.82%) in 6556.12 seconds.
    Processed 9700000/12497500 comparisons (77.62%) in 6623.81 seconds.
    Processed 9800000/12497500 comparisons (78.42%) in 6691.47 seconds.
    Processed 9900000/12497500 comparisons (79.22%) in 6759.07 seconds.
    Processed 10000000/12497500 comparisons (80.02%) in 6826.67 seconds.
    Processed 10100000/12497500 comparisons (80.82%) in 6894.81 seconds.
    Processed 10200000/12497500 comparisons (81.62%) in 6962.39 seconds.
    Processed 10300000/12497500 comparisons (82.42%) in 7030.24 seconds.
    Processed 10400000/12497500 comparisons (83.22%) in 7098.65 seconds.
    Processed 10500000/12497500 comparisons (84.02%) in 7170.07 seconds.
    Processed 10600000/12497500 comparisons (84.82%) in 7238.70 seconds.
    Processed 10700000/12497500 comparisons (85.62%) in 7308.65 seconds.
    Processed 10800000/12497500 comparisons (86.42%) in 7380.07 seconds.
    Processed 10900000/12497500 comparisons (87.22%) in 7450.55 seconds.
    Processed 11000000/12497500 comparisons (88.02%) in 7520.10 seconds.
    Processed 11100000/12497500 comparisons (88.82%) in 7589.53 seconds.
    Processed 11200000/12497500 comparisons (89.62%) in 7660.04 seconds.
    Processed 11300000/12497500 comparisons (90.42%) in 7729.92 seconds.
    Processed 11400000/12497500 comparisons (91.22%) in 7800.03 seconds.
    Processed 11500000/12497500 comparisons (92.02%) in 7873.22 seconds.
    Processed 11600000/12497500 comparisons (92.82%) in 7944.40 seconds.
    Processed 11700000/12497500 comparisons (93.62%) in 8015.84 seconds.
    Processed 11800000/12497500 comparisons (94.42%) in 8086.76 seconds.
    Processed 11900000/12497500 comparisons (95.22%) in 8160.53 seconds.
    Processed 12000000/12497500 comparisons (96.02%) in 8230.96 seconds.
    Processed 12100000/12497500 comparisons (96.82%) in 8300.28 seconds.
    Processed 12200000/12497500 comparisons (97.62%) in 8370.64 seconds.
    Processed 12300000/12497500 comparisons (98.42%) in 8439.52 seconds.
    Processed 12400000/12497500 comparisons (99.22%) in 8507.69 seconds.
    Edges added in 8577.37 seconds.
    


```python

nx.draw(G_7, node_size=10, with_labels=False)
plt.show()
```


    
![png](/assets/reutersCorpus_files/reutersCorpus_40_0.png)
    



```python
print(f"Total number of edges in the graph: {G_7.number_of_edges()}")

print("\nFirst 100 edge weights:")
for i, (u, v, attr) in enumerate(G_7.edges(data=True)):
    if i < 100: 
        print(f"Edge ({u}, {v}) - Weight: {attr['weight']:.4f}")
    else:
        break
```

    Total number of edges in the graph: 5171
    
    First 100 edge weights:
    Edge (3, 4) - Weight: 0.9272
    Edge (9, 11) - Weight: 0.9984
    Edge (14, 47) - Weight: 0.7041
    Edge (18, 23) - Weight: 0.7034
    Edge (19, 23) - Weight: 0.7906
    Edge (19, 24) - Weight: 0.9440
    Edge (21, 22) - Weight: 0.9381
    Edge (23, 24) - Weight: 0.8176
    Edge (30, 47) - Weight: 0.8200
    Edge (31, 32) - Weight: 0.9923
    Edge (36, 37) - Weight: 0.9708
    Edge (41, 43) - Weight: 0.9785
    Edge (41, 44) - Weight: 0.9988
    Edge (43, 44) - Weight: 0.9795
    Edge (51, 54) - Weight: 0.9941
    Edge (52, 53) - Weight: 0.9253
    Edge (59, 61) - Weight: 0.7193
    Edge (60, 61) - Weight: 0.7503
    Edge (61, 62) - Weight: 0.7838
    Edge (63, 64) - Weight: 0.8625
    Edge (63, 65) - Weight: 0.9621
    Edge (64, 65) - Weight: 0.8641
    Edge (68, 69) - Weight: 0.9845
    Edge (70, 71) - Weight: 0.9627
    Edge (70, 2568) - Weight: 0.8139
    Edge (70, 2571) - Weight: 0.8094
    Edge (71, 2568) - Weight: 0.7840
    Edge (71, 2571) - Weight: 0.7854
    Edge (72, 73) - Weight: 0.9265
    Edge (72, 96) - Weight: 0.7567
    Edge (72, 97) - Weight: 0.7560
    Edge (73, 96) - Weight: 0.7594
    Edge (73, 97) - Weight: 0.7641
    Edge (74, 75) - Weight: 0.9882
    Edge (76, 77) - Weight: 0.9007
    Edge (78, 705) - Weight: 0.9207
    Edge (80, 81) - Weight: 0.9894
    Edge (82, 83) - Weight: 0.8581
    Edge (82, 89) - Weight: 0.7236
    Edge (82, 725) - Weight: 0.7771
    Edge (82, 726) - Weight: 0.8489
    Edge (82, 869) - Weight: 0.7088
    Edge (83, 89) - Weight: 0.7268
    Edge (83, 725) - Weight: 0.7817
    Edge (83, 726) - Weight: 0.8084
    Edge (84, 86) - Weight: 0.7245
    Edge (84, 87) - Weight: 0.7267
    Edge (86, 87) - Weight: 0.9873
    Edge (89, 90) - Weight: 0.8252
    Edge (89, 722) - Weight: 0.7348
    Edge (89, 725) - Weight: 0.7240
    Edge (89, 726) - Weight: 0.7161
    Edge (89, 876) - Weight: 0.7087
    Edge (90, 876) - Weight: 0.8034
    Edge (91, 93) - Weight: 0.7022
    Edge (91, 94) - Weight: 0.7130
    Edge (91, 95) - Weight: 0.7076
    Edge (93, 94) - Weight: 0.9846
    Edge (93, 95) - Weight: 0.9933
    Edge (94, 95) - Weight: 0.9914
    Edge (96, 97) - Weight: 0.9808
    Edge (98, 99) - Weight: 0.9398
    Edge (98, 2550) - Weight: 0.9904
    Edge (99, 2550) - Weight: 0.9515
    Edge (100, 101) - Weight: 0.9214
    Edge (103, 104) - Weight: 0.7430
    Edge (103, 105) - Weight: 0.8277
    Edge (103, 106) - Weight: 0.8381
    Edge (103, 107) - Weight: 0.7414
    Edge (103, 959) - Weight: 0.7785
    Edge (104, 105) - Weight: 0.8113
    Edge (104, 106) - Weight: 0.8572
    Edge (104, 107) - Weight: 0.9999
    Edge (105, 106) - Weight: 0.9444
    Edge (105, 107) - Weight: 0.8096
    Edge (106, 107) - Weight: 0.8554
    Edge (108, 109) - Weight: 0.9174
    Edge (108, 3310) - Weight: 0.7165
    Edge (109, 3310) - Weight: 0.7140
    Edge (111, 120) - Weight: 0.7723
    Edge (111, 121) - Weight: 0.7740
    Edge (112, 113) - Weight: 0.9279
    Edge (112, 114) - Weight: 0.9672
    Edge (112, 115) - Weight: 0.9041
    Edge (113, 114) - Weight: 0.9065
    Edge (113, 115) - Weight: 0.9243
    Edge (114, 115) - Weight: 0.9303
    Edge (116, 119) - Weight: 0.7781
    Edge (116, 124) - Weight: 0.7052
    Edge (116, 134) - Weight: 0.7402
    Edge (119, 124) - Weight: 0.7628
    Edge (119, 134) - Weight: 0.8235
    Edge (120, 121) - Weight: 0.9881
    Edge (122, 123) - Weight: 0.9359
    Edge (122, 2635) - Weight: 0.7331
    Edge (123, 2635) - Weight: 0.7858
    Edge (124, 134) - Weight: 0.7412
    Edge (126, 127) - Weight: 0.8737
    Edge (126, 128) - Weight: 1.0000
    Edge (127, 128) - Weight: 0.8737
    


```python
from torch_geometric.utils import from_networkx
import torch


data = from_networkx(G_7)


data.x = torch.tensor(features.todense(), dtype=torch.float)
data.y = torch.tensor(labels, dtype=torch.long)


data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.train_mask[:len(train_texts)] = 1
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.test_mask[len(train_texts):] = 1

```


```python
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.log_softmax(x, dim=1)

```


```python

input_dim = data.num_features
hidden_dim = 64
output_dim = len(label_encoder.classes_)


model = GCN(input_dim, hidden_dim, output_dim)


optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

```


```python

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test():
    model.eval()
    out = model(data)
    _, top5_pred = out.topk(5, dim=1)  
    
    correct = int(top5_pred[data.test_mask].eq(data.y[data.test_mask].view(-1, 1)).sum().item())
    acc = correct / int(data.test_mask.sum())
    
   
    _, pred = out.max(dim=1)
    top1_correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    top1_acc = top1_correct / int(data.test_mask.sum())
    
    return top1_acc, acc, pred


```

### GCN Threshold .7 Results


```python

epochs = 500  
best_top1_acc = 0  
best_top5_acc = 0  

for epoch in range(epochs):
    loss = train()
    if epoch % 10 == 0:
        top1_acc, top5_acc, _ = test()
        print(f'Epoch: {epoch}, Loss: {loss:.4f}, Top-1 Accuracy: {top1_acc:.4f}, Top-5 Accuracy: {top5_acc:.4f}')
        
      
        if top1_acc > best_top1_acc:
            best_top1_acc = top1_acc
            torch.save(model.state_dict(), 'best_model_top1.pth')
        if top5_acc > best_top5_acc:
            best_top5_acc = top5_acc
            torch.save(model.state_dict(), 'best_model_top5.pth')


top1_acc, top5_acc, pred = test()
print(f'Final Top-1 Accuracy: {top1_acc:.4f}')
print(f'Final Top-5 Accuracy: {top5_acc:.4f}')


```

    Epoch: 0, Loss: 0.2965, Top-1 Accuracy: 0.6836, Top-5 Accuracy: 0.9640
    Epoch: 10, Loss: 0.2944, Top-1 Accuracy: 0.6792, Top-5 Accuracy: 0.9640
    Epoch: 20, Loss: 0.3001, Top-1 Accuracy: 0.6860, Top-5 Accuracy: 0.9676
    Epoch: 30, Loss: 0.3001, Top-1 Accuracy: 0.6844, Top-5 Accuracy: 0.9640
    Epoch: 40, Loss: 0.3003, Top-1 Accuracy: 0.6844, Top-5 Accuracy: 0.9648
    Epoch: 50, Loss: 0.2950, Top-1 Accuracy: 0.6840, Top-5 Accuracy: 0.9644
    Epoch: 60, Loss: 0.2924, Top-1 Accuracy: 0.6848, Top-5 Accuracy: 0.9644
    Epoch: 70, Loss: 0.2932, Top-1 Accuracy: 0.6812, Top-5 Accuracy: 0.9660
    Epoch: 80, Loss: 0.2911, Top-1 Accuracy: 0.6844, Top-5 Accuracy: 0.9664
    Epoch: 90, Loss: 0.2924, Top-1 Accuracy: 0.6764, Top-5 Accuracy: 0.9644
    Epoch: 100, Loss: 0.2951, Top-1 Accuracy: 0.6812, Top-5 Accuracy: 0.9640
    Epoch: 110, Loss: 0.2925, Top-1 Accuracy: 0.6848, Top-5 Accuracy: 0.9636
    Epoch: 120, Loss: 0.3018, Top-1 Accuracy: 0.6812, Top-5 Accuracy: 0.9656
    Epoch: 130, Loss: 0.2956, Top-1 Accuracy: 0.6756, Top-5 Accuracy: 0.9644
    Epoch: 140, Loss: 0.2879, Top-1 Accuracy: 0.6808, Top-5 Accuracy: 0.9640
    Epoch: 150, Loss: 0.2936, Top-1 Accuracy: 0.6828, Top-5 Accuracy: 0.9644
    Epoch: 160, Loss: 0.2992, Top-1 Accuracy: 0.6840, Top-5 Accuracy: 0.9664
    Epoch: 170, Loss: 0.3014, Top-1 Accuracy: 0.6832, Top-5 Accuracy: 0.9652
    Epoch: 180, Loss: 0.2854, Top-1 Accuracy: 0.6864, Top-5 Accuracy: 0.9664
    Epoch: 190, Loss: 0.3044, Top-1 Accuracy: 0.6820, Top-5 Accuracy: 0.9648
    Epoch: 200, Loss: 0.2945, Top-1 Accuracy: 0.6848, Top-5 Accuracy: 0.9660
    Epoch: 210, Loss: 0.2958, Top-1 Accuracy: 0.6876, Top-5 Accuracy: 0.9656
    Epoch: 220, Loss: 0.3005, Top-1 Accuracy: 0.6856, Top-5 Accuracy: 0.9620
    Epoch: 230, Loss: 0.2896, Top-1 Accuracy: 0.6856, Top-5 Accuracy: 0.9664
    Epoch: 240, Loss: 0.3054, Top-1 Accuracy: 0.6856, Top-5 Accuracy: 0.9644
    Epoch: 250, Loss: 0.2910, Top-1 Accuracy: 0.6840, Top-5 Accuracy: 0.9644
    Epoch: 260, Loss: 0.2983, Top-1 Accuracy: 0.6824, Top-5 Accuracy: 0.9672
    Epoch: 270, Loss: 0.2868, Top-1 Accuracy: 0.6824, Top-5 Accuracy: 0.9660
    Epoch: 280, Loss: 0.3035, Top-1 Accuracy: 0.6892, Top-5 Accuracy: 0.9644
    Epoch: 290, Loss: 0.3023, Top-1 Accuracy: 0.6872, Top-5 Accuracy: 0.9632
    Epoch: 300, Loss: 0.2953, Top-1 Accuracy: 0.6812, Top-5 Accuracy: 0.9672
    Epoch: 310, Loss: 0.3017, Top-1 Accuracy: 0.6844, Top-5 Accuracy: 0.9652
    Epoch: 320, Loss: 0.2766, Top-1 Accuracy: 0.6824, Top-5 Accuracy: 0.9660
    Epoch: 330, Loss: 0.2905, Top-1 Accuracy: 0.6784, Top-5 Accuracy: 0.9644
    Epoch: 340, Loss: 0.2877, Top-1 Accuracy: 0.6848, Top-5 Accuracy: 0.9660
    Epoch: 350, Loss: 0.2998, Top-1 Accuracy: 0.6764, Top-5 Accuracy: 0.9644
    Epoch: 360, Loss: 0.2853, Top-1 Accuracy: 0.6800, Top-5 Accuracy: 0.9644
    Epoch: 370, Loss: 0.2964, Top-1 Accuracy: 0.6848, Top-5 Accuracy: 0.9668
    Epoch: 380, Loss: 0.2906, Top-1 Accuracy: 0.6776, Top-5 Accuracy: 0.9672
    Epoch: 390, Loss: 0.2920, Top-1 Accuracy: 0.6896, Top-5 Accuracy: 0.9644
    Epoch: 400, Loss: 0.2932, Top-1 Accuracy: 0.6824, Top-5 Accuracy: 0.9644
    Epoch: 410, Loss: 0.2926, Top-1 Accuracy: 0.6924, Top-5 Accuracy: 0.9664
    Epoch: 420, Loss: 0.2965, Top-1 Accuracy: 0.6824, Top-5 Accuracy: 0.9648
    Epoch: 430, Loss: 0.2980, Top-1 Accuracy: 0.6780, Top-5 Accuracy: 0.9640
    Epoch: 440, Loss: 0.2921, Top-1 Accuracy: 0.6860, Top-5 Accuracy: 0.9656
    Epoch: 450, Loss: 0.2949, Top-1 Accuracy: 0.6796, Top-5 Accuracy: 0.9652
    Epoch: 460, Loss: 0.2950, Top-1 Accuracy: 0.6796, Top-5 Accuracy: 0.9668
    Epoch: 470, Loss: 0.3037, Top-1 Accuracy: 0.6828, Top-5 Accuracy: 0.9640
    Epoch: 480, Loss: 0.2976, Top-1 Accuracy: 0.6836, Top-5 Accuracy: 0.9668
    Epoch: 490, Loss: 0.3028, Top-1 Accuracy: 0.6808, Top-5 Accuracy: 0.9676
    Final Top-1 Accuracy: 0.6816
    Final Top-5 Accuracy: 0.9640
    


```python

top1_acc, top5_acc, pred = test()
print(f'Final Top-1 Accuracy: {top1_acc:.4f}')
print(f'Final Top-5 Accuracy: {top5_acc:.4f}')


author_correct = np.zeros(len(label_encoder.classes_))
author_total = np.zeros(len(label_encoder.classes_))

for i in range(len(data.y[data.test_mask])):
    label = data.y[data.test_mask][i].item()
    author_total[label] += 1
    if pred[data.test_mask][i].item() == label:
        author_correct[label] += 1

author_accuracy = author_correct / author_total

plt.figure(figsize=(12, 6))
plt.bar(range(len(label_encoder.classes_)), author_accuracy, tick_label=label_encoder.classes_)
plt.xlabel('Authors')
plt.ylabel('Top-1 Accuracy')
plt.title('Top-1 Accuracy by Author')
plt.xticks(rotation=90)
plt.show()


cm = confusion_matrix(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


top5_author_correct = np.zeros(len(label_encoder.classes_))
top5_pred = model(data).topk(5, dim=1)[1]

for i in range(len(data.y[data.test_mask])):
    label = data.y[data.test_mask][i].item()
    if label in top5_pred[data.test_mask][i]:
        top5_author_correct[label] += 1

top5_author_accuracy = top5_author_correct / author_total

plt.figure(figsize=(12, 6))
plt.bar(range(len(label_encoder.classes_)), top5_author_accuracy, tick_label=label_encoder.classes_)
plt.xlabel('Authors')
plt.ylabel('Top-5 Accuracy')
plt.title('Top-5 Accuracy by Author')
plt.xticks(rotation=90)
plt.show()

```

    Final Top-1 Accuracy: 0.6816
    Final Top-5 Accuracy: 0.9640
    


    
![png](/assets/reutersCorpus_files/reutersCorpus_48_1.png)
    



    
![png](/assets/reutersCorpus_files/reutersCorpus_48_2.png)
    



    
![png](/assets/reutersCorpus_files/reutersCorpus_48_3.png)
    


### BERT Embedding Extraction
This code uses BERT (Bidirectional Encoder Representations from Transformers) to generate embeddings for each document. The BertTokenizer tokenizes the text, and the BertModel extracts embeddings from the tokenized input. The embeddings are calculated by averaging the hidden states from BERT’s output across all tokens. Each document’s embedding is stored in a NumPy array for further processing.


```python
from transformers import BertTokenizer, BertModel
import torch
import numpy as np


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding


document_embeddings = np.array([get_bert_embedding(doc) for doc in all_texts])

```

### Graph Construction with BERT Embeddings
This code creates a graph where each node represents a document's BERT embedding. Edges between nodes are added based on the cosine similarity between document embeddings, with a threshold of 0.9. The script iterates over all document pairs, adding edges for pairs with similarity exceeding the threshold. Progress updates and timing are provided to track the performance of node and edge addition.


```python
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import time


G_bert = nx.Graph()


print("Adding nodes to the graph...")
start_time = time.time()

for i in range(document_embeddings.shape[0]):
    G_bert.add_node(i, feature=document_embeddings[i], label=labels[i])

end_time = time.time()
print(f"Nodes added: {document_embeddings.shape[0]} nodes in {end_time - start_time:.2f} seconds.")


threshold = 0.9  
print("Adding edges to the graph based on cosine similarity...")
start_time = time.time()

total_comparisons = (document_embeddings.shape[0] * (document_embeddings.shape[0] - 1)) // 2
comparison_count = 0

for i in range(document_embeddings.shape[0]):
    for j in range(i + 1, document_embeddings.shape[0]):
        similarity = cosine_similarity([document_embeddings[i]], [document_embeddings[j]])[0][0]
        if similarity > threshold:
            G_bert.add_edge(i, j, weight=similarity)
        comparison_count += 1
        
       
        if comparison_count % 100000 == 0:
            elapsed_time = time.time() - start_time
            print(f"Processed {comparison_count}/{total_comparisons} comparisons "
                  f"({(comparison_count / total_comparisons) * 100:.2f}%) "
                  f"in {elapsed_time:.2f} seconds.")

end_time = time.time()
print(f"Edges added in {end_time - start_time:.2f} seconds.")

```

    Adding nodes to the graph...
    Nodes added: 1000 nodes in 0.00 seconds.
    Adding edges to the graph based on cosine similarity...
    Processed 100000/499500 comparisons (20.02%) in 34.35 seconds.
    Processed 200000/499500 comparisons (40.04%) in 68.25 seconds.
    Processed 300000/499500 comparisons (60.06%) in 104.33 seconds.
    Processed 400000/499500 comparisons (80.08%) in 138.85 seconds.
    Edges added in 172.99 seconds.
    


```python
from torch_geometric.utils import from_networkx
import torch


data = from_networkx(G_bert)
data.x = torch.tensor(document_embeddings, dtype=torch.float)
data.y = torch.tensor(labels, dtype=torch.long)


data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.train_mask[:len(train_texts)] = 1
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.test_mask[len(train_texts):] = 1

```

### BERT GCN Results


```python
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.log_softmax(x, dim=1)


input_dim = data.num_features
hidden_dim = 64
output_dim = len(label_encoder.classes_)


model = GCN(input_dim, hidden_dim, output_dim)


optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test():
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())
    return acc


epochs = 1000  
for epoch in range(epochs):
    loss = train()
    if epoch % 10 == 0:
        test_acc = test()
        print(f'Epoch: {epoch}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f}')


test_acc = test()
print(f'Final Test Accuracy: {test_acc:.4f}')

```

    Epoch: 0, Loss: 3.9408, Test Accuracy: 0.0200
    Epoch: 10, Loss: 3.7798, Test Accuracy: 0.0600
    Epoch: 20, Loss: 3.5137, Test Accuracy: 0.0740
    Epoch: 30, Loss: 3.3142, Test Accuracy: 0.0980
    Epoch: 40, Loss: 3.1085, Test Accuracy: 0.1340
    Epoch: 50, Loss: 2.9346, Test Accuracy: 0.1520
    Epoch: 60, Loss: 2.8067, Test Accuracy: 0.1600
    Epoch: 70, Loss: 2.7456, Test Accuracy: 0.1760
    Epoch: 80, Loss: 2.6634, Test Accuracy: 0.1640
    Epoch: 90, Loss: 2.6515, Test Accuracy: 0.1860
    Epoch: 100, Loss: 2.5887, Test Accuracy: 0.1940
    Epoch: 110, Loss: 2.5755, Test Accuracy: 0.1980
    Epoch: 120, Loss: 2.5023, Test Accuracy: 0.1900
    Epoch: 130, Loss: 2.4617, Test Accuracy: 0.1960
    Epoch: 140, Loss: 2.4309, Test Accuracy: 0.1920
    Epoch: 150, Loss: 2.5300, Test Accuracy: 0.1760
    Epoch: 160, Loss: 2.4635, Test Accuracy: 0.2060
    Epoch: 170, Loss: 2.3769, Test Accuracy: 0.2060
    Epoch: 180, Loss: 2.3714, Test Accuracy: 0.1980
    Epoch: 190, Loss: 2.3119, Test Accuracy: 0.2160
    Epoch: 200, Loss: 2.3567, Test Accuracy: 0.2080
    Epoch: 210, Loss: 2.4224, Test Accuracy: 0.2060
    Epoch: 220, Loss: 2.2841, Test Accuracy: 0.1980
    Epoch: 230, Loss: 2.4662, Test Accuracy: 0.2220
    Epoch: 240, Loss: 2.2861, Test Accuracy: 0.2220
    Epoch: 250, Loss: 2.3371, Test Accuracy: 0.1980
    Epoch: 260, Loss: 2.2668, Test Accuracy: 0.2280
    Epoch: 270, Loss: 2.2233, Test Accuracy: 0.2220
    Epoch: 280, Loss: 2.2346, Test Accuracy: 0.2180
    Epoch: 290, Loss: 2.2375, Test Accuracy: 0.2060
    Epoch: 300, Loss: 2.2932, Test Accuracy: 0.2060
    Epoch: 310, Loss: 2.2285, Test Accuracy: 0.2180
    Epoch: 320, Loss: 2.3099, Test Accuracy: 0.2120
    Epoch: 330, Loss: 2.1890, Test Accuracy: 0.2160
    Epoch: 340, Loss: 2.1859, Test Accuracy: 0.2220
    Epoch: 350, Loss: 2.2381, Test Accuracy: 0.2100
    Epoch: 360, Loss: 2.2524, Test Accuracy: 0.2020
    Epoch: 370, Loss: 2.2568, Test Accuracy: 0.2160
    Epoch: 380, Loss: 2.2141, Test Accuracy: 0.2300
    Epoch: 390, Loss: 2.1450, Test Accuracy: 0.2180
    Epoch: 400, Loss: 2.2236, Test Accuracy: 0.2060
    Epoch: 410, Loss: 2.1459, Test Accuracy: 0.2360
    Epoch: 420, Loss: 2.1179, Test Accuracy: 0.2200
    Epoch: 430, Loss: 2.1261, Test Accuracy: 0.2380
    Epoch: 440, Loss: 2.1637, Test Accuracy: 0.2060
    Epoch: 450, Loss: 2.1233, Test Accuracy: 0.2300
    Epoch: 460, Loss: 2.1362, Test Accuracy: 0.2180
    Epoch: 470, Loss: 2.1053, Test Accuracy: 0.2320
    Epoch: 480, Loss: 2.1502, Test Accuracy: 0.2480
    Epoch: 490, Loss: 2.1838, Test Accuracy: 0.2240
    Epoch: 500, Loss: 2.1493, Test Accuracy: 0.2380
    Epoch: 510, Loss: 2.0812, Test Accuracy: 0.2380
    Epoch: 520, Loss: 2.1669, Test Accuracy: 0.2200
    Epoch: 530, Loss: 2.1022, Test Accuracy: 0.2420
    Epoch: 540, Loss: 2.0982, Test Accuracy: 0.2320
    Epoch: 550, Loss: 2.0515, Test Accuracy: 0.2480
    Epoch: 560, Loss: 2.0752, Test Accuracy: 0.2400
    Epoch: 570, Loss: 2.1334, Test Accuracy: 0.2400
    Epoch: 580, Loss: 2.0811, Test Accuracy: 0.2400
    Epoch: 590, Loss: 2.0972, Test Accuracy: 0.2520
    Epoch: 600, Loss: 2.0987, Test Accuracy: 0.2400
    Epoch: 610, Loss: 2.0783, Test Accuracy: 0.2180
    Epoch: 620, Loss: 2.2056, Test Accuracy: 0.2280
    Epoch: 630, Loss: 2.1457, Test Accuracy: 0.2220
    Epoch: 640, Loss: 2.0310, Test Accuracy: 0.2300
    Epoch: 650, Loss: 2.0460, Test Accuracy: 0.2440
    Epoch: 660, Loss: 2.0344, Test Accuracy: 0.2440
    Epoch: 670, Loss: 2.2894, Test Accuracy: 0.2580
    Epoch: 680, Loss: 2.0254, Test Accuracy: 0.2360
    Epoch: 690, Loss: 2.0207, Test Accuracy: 0.2480
    Epoch: 700, Loss: 2.0512, Test Accuracy: 0.2480
    Epoch: 710, Loss: 2.1853, Test Accuracy: 0.2420
    Epoch: 720, Loss: 2.0179, Test Accuracy: 0.2320
    Epoch: 730, Loss: 2.1235, Test Accuracy: 0.2560
    Epoch: 740, Loss: 2.0206, Test Accuracy: 0.2320
    Epoch: 750, Loss: 2.0675, Test Accuracy: 0.2680
    Epoch: 760, Loss: 2.0031, Test Accuracy: 0.2420
    Epoch: 770, Loss: 1.9965, Test Accuracy: 0.2380
    Epoch: 780, Loss: 2.1080, Test Accuracy: 0.2160
    Epoch: 790, Loss: 2.0693, Test Accuracy: 0.2440
    Epoch: 800, Loss: 1.9877, Test Accuracy: 0.2280
    Epoch: 810, Loss: 1.9826, Test Accuracy: 0.2500
    Epoch: 820, Loss: 1.9904, Test Accuracy: 0.2620
    Epoch: 830, Loss: 2.0275, Test Accuracy: 0.2300
    Epoch: 840, Loss: 2.0756, Test Accuracy: 0.2060
    Epoch: 850, Loss: 1.9445, Test Accuracy: 0.2520
    Epoch: 860, Loss: 1.9844, Test Accuracy: 0.2500
    Epoch: 870, Loss: 2.1021, Test Accuracy: 0.2200
    Epoch: 880, Loss: 2.0136, Test Accuracy: 0.2340
    Epoch: 890, Loss: 1.9633, Test Accuracy: 0.2360
    Epoch: 900, Loss: 1.9825, Test Accuracy: 0.2480
    Epoch: 910, Loss: 1.9587, Test Accuracy: 0.2600
    Epoch: 920, Loss: 2.1704, Test Accuracy: 0.2540
    Epoch: 930, Loss: 2.0696, Test Accuracy: 0.2700
    Epoch: 940, Loss: 1.9428, Test Accuracy: 0.2300
    Epoch: 950, Loss: 2.0681, Test Accuracy: 0.2560
    Epoch: 960, Loss: 2.0858, Test Accuracy: 0.2540
    Epoch: 970, Loss: 1.9424, Test Accuracy: 0.2380
    Epoch: 980, Loss: 2.0104, Test Accuracy: 0.2500
    Epoch: 990, Loss: 1.9684, Test Accuracy: 0.2480
    Final Test Accuracy: 0.2620
    


```python
print(f"Total number of edges in the graph: {G_bert.number_of_edges()}")

print("\nFirst 100 edge weights:")
for i, (u, v, attr) in enumerate(G_bert.edges(data=True)):
    if i < 100:  
        print(f"Edge ({u}, {v}) - Weight: {attr['weight']:.4f}")
    else:
        break
```

    Total number of edges in the graph: 139659
    
    First 100 edge weights:
    Edge (0, 1) - Weight: 0.9086
    Edge (0, 2) - Weight: 0.9264
    Edge (0, 4) - Weight: 0.9279
    Edge (0, 5) - Weight: 0.9119
    Edge (0, 6) - Weight: 0.9216
    Edge (0, 7) - Weight: 0.9227
    Edge (0, 8) - Weight: 0.9317
    Edge (0, 9) - Weight: 0.9144
    Edge (0, 52) - Weight: 0.9187
    Edge (0, 75) - Weight: 0.9026
    Edge (0, 76) - Weight: 0.9010
    Edge (0, 79) - Weight: 0.9001
    Edge (0, 91) - Weight: 0.9060
    Edge (0, 93) - Weight: 0.9233
    Edge (0, 104) - Weight: 0.9041
    Edge (0, 210) - Weight: 0.9108
    Edge (0, 220) - Weight: 0.9321
    Edge (0, 221) - Weight: 0.9095
    Edge (0, 223) - Weight: 0.9130
    Edge (0, 225) - Weight: 0.9179
    Edge (0, 228) - Weight: 0.9041
    Edge (0, 242) - Weight: 0.9014
    Edge (0, 246) - Weight: 0.9010
    Edge (0, 249) - Weight: 0.9012
    Edge (0, 280) - Weight: 0.9020
    Edge (0, 284) - Weight: 0.9027
    Edge (0, 311) - Weight: 0.9036
    Edge (0, 330) - Weight: 0.9064
    Edge (0, 331) - Weight: 0.9050
    Edge (0, 334) - Weight: 0.9070
    Edge (0, 335) - Weight: 0.9160
    Edge (0, 350) - Weight: 0.9014
    Edge (0, 351) - Weight: 0.9301
    Edge (0, 353) - Weight: 0.9355
    Edge (0, 354) - Weight: 0.9171
    Edge (0, 355) - Weight: 0.9026
    Edge (0, 358) - Weight: 0.9099
    Edge (0, 359) - Weight: 0.9311
    Edge (0, 360) - Weight: 0.9153
    Edge (0, 362) - Weight: 0.9119
    Edge (0, 368) - Weight: 0.9009
    Edge (0, 388) - Weight: 0.9044
    Edge (0, 392) - Weight: 0.9091
    Edge (0, 394) - Weight: 0.9003
    Edge (0, 396) - Weight: 0.9069
    Edge (0, 397) - Weight: 0.9130
    Edge (0, 400) - Weight: 0.9508
    Edge (0, 401) - Weight: 0.9596
    Edge (0, 403) - Weight: 0.9461
    Edge (0, 405) - Weight: 0.9584
    Edge (0, 406) - Weight: 0.9310
    Edge (0, 407) - Weight: 0.9195
    Edge (0, 408) - Weight: 0.9571
    Edge (0, 409) - Weight: 0.9591
    Edge (0, 411) - Weight: 0.9021
    Edge (0, 423) - Weight: 0.9081
    Edge (0, 425) - Weight: 0.9263
    Edge (0, 444) - Weight: 0.9064
    Edge (0, 462) - Weight: 0.9097
    Edge (0, 463) - Weight: 0.9085
    Edge (0, 485) - Weight: 0.9124
    Edge (0, 487) - Weight: 0.9005
    Edge (0, 500) - Weight: 0.9088
    Edge (0, 501) - Weight: 0.9103
    Edge (0, 502) - Weight: 0.9037
    Edge (0, 503) - Weight: 0.9269
    Edge (0, 504) - Weight: 0.9210
    Edge (0, 506) - Weight: 0.9170
    Edge (0, 508) - Weight: 0.9161
    Edge (0, 509) - Weight: 0.9027
    Edge (0, 532) - Weight: 0.9034
    Edge (0, 553) - Weight: 0.9187
    Edge (0, 554) - Weight: 0.9040
    Edge (0, 555) - Weight: 0.9059
    Edge (0, 557) - Weight: 0.9021
    Edge (0, 584) - Weight: 0.9002
    Edge (0, 590) - Weight: 0.9106
    Edge (0, 591) - Weight: 0.9460
    Edge (0, 592) - Weight: 0.9188
    Edge (0, 593) - Weight: 0.9026
    Edge (0, 594) - Weight: 0.9377
    Edge (0, 596) - Weight: 0.9136
    Edge (0, 720) - Weight: 0.9079
    Edge (0, 724) - Weight: 0.9027
    Edge (0, 733) - Weight: 0.9171
    Edge (0, 739) - Weight: 0.9181
    Edge (0, 740) - Weight: 0.9011
    Edge (0, 751) - Weight: 0.9041
    Edge (0, 795) - Weight: 0.9086
    Edge (0, 830) - Weight: 0.9127
    Edge (0, 831) - Weight: 0.9150
    Edge (0, 832) - Weight: 0.9096
    Edge (0, 833) - Weight: 0.9191
    Edge (0, 838) - Weight: 0.9122
    Edge (0, 850) - Weight: 0.9163
    Edge (0, 851) - Weight: 0.9086
    Edge (0, 852) - Weight: 0.9320
    Edge (0, 853) - Weight: 0.9167
    Edge (0, 854) - Weight: 0.9234
    Edge (0, 855) - Weight: 0.9043
    

### SVM Classifier with Top-5 Accuracy and Confusion Matrix
This code trains a Support Vector Machine (SVM) model on the TF-IDF features of the Reuters C50 dataset. The SVM is trained with a linear kernel and probability estimates enabled. After training, it evaluates the model using:

Top-1 Accuracy: The percentage of test samples where the correct author is the top predicted label.
Top-5 Accuracy: The percentage of test samples where the correct author is among the top 5 predicted labels.
Confusion Matrix: Visualizes the prediction performance across all authors.
Classification Report: Provides precision, recall, and F1-score metrics for each author.



```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

train_features = features[:len(train_texts)]
test_features = features[len(train_texts):]
train_labels = labels[:len(train_texts)]
test_labels = labels[len(test_texts):]

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np


train_features_dense = np.asarray(train_features.todense())
test_features_dense = np.asarray(test_features.todense())


svm_model = SVC(kernel='linear', probability=True)


svm_model.fit(train_features_dense, train_labels)

test_proba = svm_model.predict_proba(test_features_dense)


top5_pred = np.argsort(test_proba, axis=1)[:, -5:]


test_pred = svm_model.predict(test_features_dense)
top1_accuracy = accuracy_score(test_labels, test_pred)
print(f'SVM Top-1 Accuracy: {top1_accuracy:.4f}')


top5_correct = 0
for i in range(len(test_labels)):
    if test_labels[i] in top5_pred[i]:
        top5_correct += 1

top5_accuracy = top5_correct / len(test_labels)
print(f'SVM Top-5 Accuracy: {top5_accuracy:.4f}')

cm = confusion_matrix(test_labels, test_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for SVM')
plt.show()

print("Classification Report:")
print(classification_report(test_labels, test_pred, target_names=label_encoder.classes_))



```

    SVM Top-1 Accuracy: 0.6580
    SVM Top-5 Accuracy: 0.9400
    


    
![png](/assets/reutersCorpus_files/reutersCorpus_58_1.png)
    


    Classification Report:
                       precision    recall  f1-score   support
    
        AaronPressman       0.85      0.92      0.88        50
           AlanCrosby       0.92      0.48      0.63        50
       AlexanderSmith       0.28      0.30      0.29        50
      BenjaminKangLim       0.46      0.22      0.30        50
        BernardHickey       0.70      0.76      0.73        50
          BradDorfman       0.52      0.92      0.66        50
     DarrenSchuettler       0.44      0.28      0.34        50
          DavidLawder       0.50      0.22      0.31        50
        EdnaFernandes       0.78      0.58      0.67        50
          EricAuchard       0.43      0.56      0.49        50
       FumikoFujisaki       1.00      0.98      0.99        50
       GrahamEarnshaw       0.78      0.92      0.84        50
     HeatherScoffield       0.34      0.42      0.38        50
           JanLopatka       0.53      0.40      0.45        50
        JaneMacartney       0.26      0.14      0.18        50
         JimGilchrist       0.94      1.00      0.97        50
       JoWinterbottom       0.91      0.80      0.85        50
             JoeOrtiz       0.50      0.82      0.62        50
         JohnMastrini       0.41      0.70      0.52        50
         JonathanBirt       0.74      0.84      0.79        50
          KarlPenhaul       0.93      1.00      0.96        50
            KeithWeir       0.73      0.86      0.79        50
       KevinDrawbaugh       0.60      0.80      0.68        50
        KevinMorrison       0.62      0.68      0.65        50
        KirstinRidley       0.91      0.58      0.71        50
    KouroshKarimkhany       0.82      0.62      0.70        50
            LydiaZajc       1.00      0.72      0.84        50
       LynneO'Donnell       0.91      0.80      0.85        50
      LynnleyBrowning       0.91      1.00      0.95        50
      MarcelMichelson       0.69      0.74      0.71        50
         MarkBendeich       0.90      0.56      0.69        50
           MartinWolk       0.80      0.64      0.71        50
         MatthewBunce       1.00      0.90      0.95        50
        MichaelConnor       0.80      0.88      0.84        50
           MureDickie       0.24      0.36      0.29        50
            NickLouth       0.80      0.80      0.80        50
      PatriciaCommins       0.79      0.62      0.70        50
        PeterHumphrey       0.60      0.74      0.66        50
           PierreTran       0.83      0.60      0.70        50
           RobinSidel       0.84      0.84      0.84        50
         RogerFillion       0.97      0.74      0.84        50
          SamuelPerry       0.59      0.64      0.62        50
         SarahDavison       0.69      0.54      0.61        50
          ScottHillis       0.22      0.32      0.26        50
          SimonCowell       0.82      0.72      0.77        50
             TanEeLyn       0.47      0.58      0.52        50
       TheresePoletti       0.83      0.78      0.80        50
           TimFarrand       0.89      0.80      0.84        50
           ToddNissen       0.37      0.42      0.39        50
         WilliamKazer       0.35      0.36      0.36        50
    
             accuracy                           0.66      2500
            macro avg       0.68      0.66      0.66      2500
         weighted avg       0.68      0.66      0.66      2500
    
    

