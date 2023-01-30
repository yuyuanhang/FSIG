# FSIG
## Compile
To compile this project in release mode, use the following command.
```zsh
make mdebug=0
```
To compile this project in debug mode, use the following command.
```zsh
make mdebug=1
```

## Run
To run this project, use the following command.
```zsh
./FSIG [data graph] [query graph] [algorithm type] [graph type]
```
For example,
```zsh
./FSIG dblp.graph 4_dense_0.graph 1 1
```
Parameters:

algorithm type: 1 is FSIG; 2 is GSI.

graph type: 0 is unlabeled graph; 1 is labeled graphs.

## Datasets
The format of graphs we support is as follows.
```zsh
t [the number of vertices] [the number of edges]
v [id] [label] [degree]
...
e [src] [dest]
...
```
