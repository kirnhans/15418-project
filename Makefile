defule:
	g++ -std=c++11 -fopenmp -O3 -g -o prf DecisionTreeRepr.h main.cpp prf.cpp
test:
	./prf graphs/tiny.graph
clean:
	rm -rf prf  *~ *.*~
