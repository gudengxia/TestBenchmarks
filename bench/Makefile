SRC = cpucycles.c speed_print.c test_speed.c
HDR = cpucycles.h speed_print.h
OQSDIR=/home/fzhang/huawei
all:bench 

bench: $(SRC) $(HDR)
	gcc -o $@ $(SRC) $(HDR) -loqs -L$(OQSDIR)/lib -I$(OQSDIR)/include

.PHONY: clean
clean:
	rm bench 

