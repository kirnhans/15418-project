EXECUTABLE := cudaRF

CU_FILES   := train.cu test.cu

CU_DEPS    :=

CC_FILES   := main.cpp

all: $(EXECUTABLE)

LOGS	   := logs

###########################################################

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall -lpthread -std=c++11
LDFLAGS=-L/usr/local/cuda/lib64/ -lcudart
NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_35


OBJS=$(OBJDIR)/main.o  $(OBJDIR)/train.o  $(OBJDIR)/test.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE) $(LOGS)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
