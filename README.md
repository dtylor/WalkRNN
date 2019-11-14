# WalkRNN

A utility module for experimenting with graph embeddings for use in language models.  See paper at http://viXra.org/abs/1910.0362  

## Usage
```python
from utilities import *
from module import *

# Load graph from csv
Gkern = load_graph_kernel_graph("./examples/data/AIDS")
y = load_graph_kernel_labels("./examples/data/AIDS")

#Learn structural signatures of each node in networkx property graph and apply to node as an attribute
#Then transform into a format prepared for WalkRNN
G, current_vocab_size = transform_graph(Gkern, params={'num_kmeans_clusters': 4, "num_pca_components": 6, "num_batch":500, 'num_att_kmeans_clusters': 5})

# Generate 20 walks from each node
walks = walk_as_string(G, componentLabels = y, params={'num_walks': 20, 'walk_length': 30})

```

See Demonstration.ipynb for more details

## Testing
Run `python3 -m unittest test.py`

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details. <br>

Third party libraries used include:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**graphwave**<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Copyright 2018 contributors at Stanford<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;https://github.com/snap-stanford/graphwave<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MIT License<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**node2vec**<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Copyright (c) 2016 Aditya Grover<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;https://github.com/aditya-grover/node2vec<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MIT License<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**fast.ai**<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Copyright 2017 onwards, fast.ai, Inc.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;https://github.com/fastai/fastai<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Apache License, Version 2.0<br>
    
Third party dataset downloaded from this site: 
https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
