CXX = /usr/bin/g++
CXXFLAGS = -O3 -fomit-frame-pointer -march=native
LDFLAGS= -lm

OBJ= cbd.o fips202.o indcpa.o kem.o ntt.o poly.o polyvec.o precomp.o reduce.o rng.o verify.o aes.o
HEADERS= api.h cbd.h fips202.h indcpa.h ntt.h params.h poly.h polyvec.h reduce.h rng.h verify.h aes.h

all: libakcn.a testbench_kem

testbench_kem: testbench_kem.cxx speed_print.cxx cpucycles.cxx
	g++ $(CXXFLAGS) -o $@ $^ -lakcn -L. -static

libakcn.a: $(OBJ)
	ar crs $@ $^
	rm $(OBJ)

$(OBJ): %.o: %.cxx
	$(CXX) $(CXXFLAGS) -c -o $@ $< 

.PHONY: clean

clean:
	rm -rf libakcn.a testbench_kem
