CXX = /usr/bin/g++
CXXFLAGS = -O3 -fomit-frame-pointer -march=native
LDFLAGS= -lm
OQSDIR=/home/fzhang/huawei
OBJ = kem.o aes.o aes256ctr.o cbd.o fips202.o fips202x4.o genmatrix.o ntt.o owcpa.o poly.o polyvec.o precomp.o randombytes.o reduce.o rng.o verify.o 
EXOBJ = KeccakP-1600-times4-SIMD256.o sha512.o sha256.o
#ASM = sha_256.o sha_512.o 
HEADERS= api.h cbd.h fips202.h fips202x4.h genmatrix.h KeccakP-1600-times4-SnP.h  owcpa.h ntt.h params.h poly.h polyvec.h reduce.h rng.h SIMD256-config.h verify.h aes.h aes256ctr.h align.h brg_endian.h

all: libaigis.a testbench

testbench: test/testbench_kem.cpp test/cpucycles.c test/speed_print.c
	g++ -o testbench_kem $^ -laigis -L. -I./ 
	#g++ -o testbench_kyber testbench_kyber.cpp -I$(OQSDIR)/include -loqs -L$(OQSDIR)/lib

libaigis.a: $(OBJ) $(EXOBJ)
	ar crs $@ $^
	

$(OBJ): %.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $< -Iinclude -I.
$(EXOBJ): %.o: keccak4x/%.c
	$(CXX) $(CXXFLAGS) -c -o $@ $<  -I.
#$(ASM): %.o: keccak4x/%.s
#	$(CXX) $(CXXFLAGS) -c -o $@ $<
.PHONY: clean

clean:
	rm -rf libaigis.a testbench_kem testbench_kyber $(OBJ) $(EXOBJ) 

