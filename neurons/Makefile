CXX = g++
CXXFLAGS = -std=c++14 -Wall -Wextra -Werror -O3
LDFLAGS = -ldl -O3

CXXFILES := $(wildcard *.cpp)
OBJFILES := $(CXXFILES:.cpp=.o)

.PHONY: all clean

all: my_neurons
clean:
	rm -f my_neurons *.o

my_neurons: $(OBJFILES)
	$(CXX) -o $@ $+ $(LDFLAGS)
%.o: %.cpp
	$(CXX) -c -o $@ $< $(CXXFLAGS)
