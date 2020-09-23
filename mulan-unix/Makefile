
CC = /usr/bin/cc
CFLAGS = -Wall -g -Wextra -march=native -mtune=native -O3 -fomit-frame-pointer 
NISTFLAGS = -O3 -fomit-frame-pointer -march=native -fPIC
OBJ = Alg.o fips202.o randombytes.o Polynomial.o io.o Rounding.o
SOURCES = 	main.c \
			Alg.c \
			fips202.c \
			mulan_randombytes.c \
			Polynomial.c \
			io.c \
			Rounding.c \

HEADERS = 	mulan_randombytes.h \
			fips202.h \
			Parameters.h \
			Alg.h \
			Polynomial.h \
			io.h \
			Rounding.h \


all: mulan libmulan.a test_speed

test_speed: libmulan.a test_speed.c cpucycles.c speed_print.c 
	$(CC) $(CFLAGS) test_speed.c cpucycles.c speed_print.c -o $@ -L. -lmulan -lm

mulan: libmulan.a test.c
	$(CC) $(CFLAGS) test.c -o test -L. -lmulan
	
libmulan.a: $(OBJ)
	ar csr $@ $^
	rm $(OBJ)
$(OBJ): %.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean

clean:
	rm -rf test test_speed libmulan.a $(OBJ)
