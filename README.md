# WalkRNN

A utility module for experimenting with graph embeddings for use in language models.

## Usage
```python
from utilities import *
from module import *

# Load graph from csv and learn structural signatures of each node and apply to node as an attribute
a = load_graph_kernel_graph("./Cuneiform")
y = load_graph_kernel_labels("./Cuneiform")

# Generate 20 walks from each node
c = walk_as_string(b[0], y, params={'num_walks': 20, 'walk_length': 30})
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
    
Third party dataset downloaded from this site: 
https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
