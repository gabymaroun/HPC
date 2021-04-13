CC = scorep	gcc
HDF5_INC=`pkg-config --cflags hdf5`
HDF5_LIB=`pkg-config --libs hdf5`
CFLAGS = -g -O2 -Wall -pedantic $(HDF5_INC)
LDFLAGS = -lm $(HDF5_LIB) 


allsrc := $(wildcard *.c)
allobj := $(patsubst %.c,%.o,$(allsrc))
maintp := $(filter tp%.c,$(allsrc))
src := $(filter-out tp%.c,$(allsrc))
obj := $(patsubst %.c,%.o,$(src))
exetp := $(patsubst %.c,%.exe,$(maintp))

all : $(exetp)

.SECONDARY: $(allobj)

%.o: %.c %.h
	$(CC) -c $(CFLAGS) $(HDF5INC) -o $@ $<
%.o: %.c
	$(CC) -c $(CFLAGS) $(HDF5INC) -o $@ $<

tp%.exe : tp%.o $(obj)
	$(CC) $(CFLAGS) -o $@ $^  $(LDFLAGS) $(HDF5LIB)

clean:
	rm -f $(allobj) $(exetp)
