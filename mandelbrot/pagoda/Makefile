include ../../common/make.config

TARGET=mand

.PHONY: default all clean

default: $(TARGET)
all: default

OBJECTS = $(patsubst %.cu, %.o, $(wildcard *.cu))

%.o: %.cu
	$(NVCC) $(CFLAGS) -I$(INC) -arch=$(ARCH) -maxrregcount $(REG) -dc $< -o $@


.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(NVCC) $(CFLAGS) -I$(INC) -arch=$(ARCH) -maxrregcount $(REG) $(OBJECTS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TARGET)
