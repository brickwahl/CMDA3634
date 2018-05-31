#can change these if files are located elsewhere
SOURCEDIR = ./.
INCLUDEDIR = ./.
OBJDIR = ./.


INCLUDES = $(wildcard $(INCLUDEDIR)/*h)
SOURCE = functions.c
OBJECTS = $(subst $(SOURCEDIR)/,$(OBJDIR)/,$(SOURCE:.c=.o))

DEPS = $(INCLUDES) 
FLAGS = -I$(INCLUDEDIR) -g -O3 
NVFLAGS = -arch=sm_60
LIBS  = -lm

CC =  gcc
NVCC =  nvcc

$(OBJDIR)/%.o:$(SOURCEDIR)/%.cu $(DEPS)
	$(CC) $(FLAGS) -o $@ $(LIBS) -c $<

all: setup encrypt decrypt decryptBSGS ompdecryptBSGS mpidecryptBSGS cudaDecryptBSGS

setup:  setup.o functions.o $(DEPS)
	$(CC) setup.o functions.o $(FLAGS) -o setup $(LIBS)

encrypt: encrypt.o functions.o $(DEPS)
	$(CC)  encrypt.o functions.o $(FLAGS) -o encrypt $(LIBS)

decrypt: decrypt.o functions.o $(DEPS)
	$(CC)  decrypt.o functions.o $(FLAGS) -o decrypt $(LIBS)

decryptBSGS: decryptBSGS.o functions.o $(DEPS)
	$(CC)  decryptBSGS.o functions.o $(FLAGS) -o decryptBSGS $(LIBS)

ompdecryptBSGS: ompdecryptBSGS.c $(DEPS)
	$(CC)  ompdecryptBSGS.c $(FLAGS) -fopenmp -o ompdecryptBSGS $(LIBS)

mpidecryptBSGS: mpidecryptBSGS.c $(DEPS)
	mpicc  mpidecryptBSGS.c $(FLAGS) -fopenmp -o mpidecryptBSGS $(LIBS)

cudaDecryptBSGS: cudaDecryptBSGS.cu functions.o $(DEPS)
	$(NVCC)  cudaDecryptBSGS.cu $(FLAGS) $(NVFLAGS) -o cudaDecryptBSGS $(LIBS)

clean:
	rm -f setup encrypt decrypt decryptBSGS ompdecryptBSGS mpidecryptBSGS cudaDecryptBSGS
	rm -f $(OBJDIR)/*.o
