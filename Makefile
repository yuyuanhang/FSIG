CC = g++
NVCC = nvcc -rdc=true
CFLAGS = -c -std=c++14
EXEFLAG = -std=c++14

ifeq ($(mdebug), 0)
    CFLAGS += -O2
    EXEFLAG += -O2
endif

ifeq ($(mdebug), 1)
    NVCC += -g -G
    CC += -g
endif

NVCC += -gencode arch=compute_75,code=sm_75 \
                 -gencode arch=compute_75,code=compute_75

objdir = ./objs/
objfile = $(objdir)graph_reader.o $(objdir)graph.o $(objdir)graph_format_transformer.o $(objdir)MMatch.o $(objdir)GSI.o

all: FSIG

FSIG: $(objfile) main/run.cc utility/inverted_index.h utility/time_recorder.h
	$(NVCC) $(EXEFLAG) -o FSIG main/run.cc $(objfile)

$(objdir)graph.o: graph/graph.cc graph/graph.h utility/defines.h
	$(CC) $(CFLAGS) graph/graph.cc -o $(objdir)graph.o

$(objdir)graph_reader.o: IO/graph_reader.cc IO/graph_reader.h utility/defines.h
	$(CC) $(CFLAGS) IO/graph_reader.cc -o $(objdir)graph_reader.o

$(objdir)graph_format_transformer.o: graph_format_transformer/graph_format_transformer.cc graph_format_transformer/graph_format_transformer.h utility/defines.h
	$(CC) $(CFLAGS) graph_format_transformer/graph_format_transformer.cc -o $(objdir)graph_format_transformer.o

$(objdir)MMatch.o: match/MMatch.cu match/MMatch.h match/GSI.h match/MMatch_kernels.cuh match/subgraph_match.h match/embedding.h utility/inverted_index.h utility/defines.h utility/time_recorder.h memory_manager/memory_manager.h
	$(NVCC) $(CFLAGS) match/MMatch.cu -o $(objdir)MMatch.o

$(objdir)GSI.o: match/GSI.cu match/GSI.h match/GSI_kernels.cuh match/subgraph_match.h match/embedding.h utility/inverted_index.h utility/defines.h utility/time_recorder.h memory_manager/memory_manager.h
	$(NVCC) $(CFLAGS) match/GSI.cu  -o $(objdir)GSI.o

.PHONY: clean dist tarball test sumlines

clean:
	rm -f $(objdir)*