CC = /usr/bin/cc
CFLAGS = -Wall -g -Wextra -march=native -mtune=native -O3 -fomit-frame-pointer 
NISTFLAGS = -O3 -fomit-frame-pointer -march=native -fPIC

OBJ = 	aes.o aes256ctr.o Alg.o fips202.o fips202x4.o ntt.o packing.o poly.o polyvec.o randombytes.o rng.o sign.o
EX_OBJ = KeccakP-1600-times4-SIMD256.o			
SOURCES = 	main.c \
			Alg.c \
			fips202.c \
			randombytes.c \
			Polynomial.c \
			io.c \
			Rounding.c \

HEADERS = 	randombytes.h \
			fips202.h \
			Parameters.h \
			Alg.h \
			Polynomial.h \
			io.h \
			Rounding.h \


all: test test_speed libaigis.a

test_speed: test_speed.c cpucycles.c speed_print.c
	$(CC) $(CFLAGS) $^ -o $@ -laigis -L. -lm
	
libaigis.a: $(OBJ) $(EX_OBJ)
	ar csr $@ $^
	rm -rf $^
$(OBJ): %.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(EX_OBJ): %.o: keccak4x/%.c
	$(CC) $(CFLAGS) -c $< -o $@
	
test: test.c api.h libaigis.a 
	$(CC) $(CFLAGS) test.c -o $@ -laigis -L.

.PHONY: clean

clean:
	rm -f test test_speed libaigis.a
