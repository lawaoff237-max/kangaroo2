# =============================================================================
#  Makefile - kangaroo (secp256k1), CPU multi-coeurs + GPU CUDA optionnel
#
#  make            : detecte nvcc (GPU+CPU si present, sinon CPU seul identique a l'original)
#  make cpu        : force un binaire CPU seul
#  make kaggle     : raccourci Kaggle (2x T4, sm_75)
#  make portable   : comme "make" mais SANS asm PTX (-DKG_PORTABLE) : plus lent, a n'utiliser
#                    que si l'auto-test arithmetique GPU signale un ecart
#  make info       : comme "make" avec -Xptxas -v (registres / spills par noyau)
#  make native     : -arch=native (CUDA >= 11.6) : cible exactement la carte presente
#  make clean
#
#  Autres cartes : make GENCODE="-gencode arch=compute_80,code=sm_80"   (A100)
#                                 compute_86 (RTX 30xx / A10), compute_89 (RTX 40xx / L4), compute_90 (H100)
#  Le lien final passe TOUJOURS par nvcc (il connait le bon chemin de libcudart).
# =============================================================================
CXX      ?= g++
NVCC     := $(shell command -v nvcc 2>/dev/null)
CXXFLAGS ?= -O3 -march=native -flto -funroll-loops -std=c++17 -pthread
EXTRA    ?=
NVCCFLAGS?= -O3 -std=c++17 -diag-suppress 20091 $(EXTRA) -Xcompiler "-O3 -pthread"
# sm_75 = Turing (T4, Kaggle) + PTX 7.5 (recompilation automatique par le pilote sur cartes plus recentes)
GENCODE  ?= -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75

BIN = kangaroo

.PHONY: all cpu kaggle portable info native clean
all:
ifeq ($(NVCC),)
	@echo "nvcc introuvable : build CPU seul (identique a l'original)."
	$(CXX) $(CXXFLAGS) kangaroo.cpp -o $(BIN)
else
	@echo "nvcc trouve ($(NVCC)) : build CPU + GPU."
	$(NVCC) $(NVCCFLAGS) $(GENCODE) -c kangaroo_gpu.cu -o kangaroo_gpu.o
	$(CXX) $(CXXFLAGS) -DHAVE_CUDA -c kangaroo.cpp -o kangaroo.o
	$(NVCC) $(NVCCFLAGS) $(GENCODE) kangaroo.o kangaroo_gpu.o -o $(BIN)
endif

cpu:
	$(CXX) $(CXXFLAGS) kangaroo.cpp -o $(BIN)

kaggle: all
	@echo "OK : ./kangaroo -gpu all  (utilise les 2 T4 + tous les coeurs CPU)"

portable:
	$(MAKE) all EXTRA="-DKG_PORTABLE"

info:
	$(MAKE) all EXTRA="-Xptxas -v"

native:
	$(MAKE) all GENCODE="-arch=native"

clean:
	rm -f $(BIN) kangaroo.o kangaroo_gpu.o
