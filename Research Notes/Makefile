CXX ?= g++
CXXFLAGS += -c -pg -Wall $(shell pkg-config --cflags opencv)  
LDFLAGS += -pg $(shell pkg-config --libs --static opencv)

all: $(NAME)

$(NAME): $(NAME).o; $(CXX) $< -o $@ $(LDFLAGS)

%.o: %.cpp; $(CXX) $< -o $@ $(CXXFLAGS)

clean: ; rm -f $(NAME).o $(NAME)
