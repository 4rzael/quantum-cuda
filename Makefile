# @Author: Julien Vial-Detambel <l3ninj>
# @Date:   2018-06-12T11:58:18+01:00
# @Email:  julien.vial-detambel@epitech.eu
# @Project: CUDA-Based Simulator of Quantum Systems
# @Filename: Makefile
# @Last modified by:   nj203
# @Last modified time: 2018-07-03T23:47:38+01:00
# @License: MIT License

CXX?=	g++
# Maccro that defines the c++ compiler we will use.

OPTIFLAGS=	-g3

CXXFLAGS=	-Wextra -Wall -std=c++14 $(OPTIFLAGS)

# Maccro that defines the Nvidia's compiler we will use.
NVCC=	nvcc

# Maccro that contains how the temporary objects and repositories will be deleted.
# E.g. here we will call the 'rm' linux command with 'rf' as options.
RM=	rm -rf

# Maccro that contains the final executable name of the project.
NAME=	quSim

# Maccro that contains where the *.cpp source files are located.
SDIR=	src

# Maccro that contains where the *.cu source files are located.
CUSDIR=	cuda_src

# Maccro that contains the repository's name where objects compiled from *.cpp
# sources files will be stored.
ODIR=	obj

# Maccro that contains the repository's name where objects compiled from *.cu
# sources files will be stored.
CUODIR=	cuda_obj

# Maccro that contains a list of all *.cu source files that will be compiled
# with 'NVCC' maccro.
CUSRC=	QCUDA.cu		\
	QCUDA_operations.cu	\
	GPUExecutor.cu		\
	CPUExecutor.cu		\
	ExecutorManager.cu

# Maccro that contains a list of all *.cpp source files that will be compiled
# with 'CXX' maccro.
SRC=	Parser/float_expr_ast.cpp \
	Parser/ASTGenerator.cpp \
	Parser/CircuitBuilder/CXBuilder.cpp \
	Parser/CircuitBuilder/MeasureBuilder.cpp \
	Parser/CircuitBuilder/RegisterDeclarationBuilder.cpp \
	Parser/CircuitBuilder/UBuilder.cpp \
	Parser/CircuitBuilder/UserDefinedGateBuilder.cpp \
	Parser/CircuitBuilder/IncludeBuilder.cpp \
	Parser/CircuitBuilder/ResetBuilder.cpp \
	Parser/CircuitBuilder/CircuitBuilder.cpp \
	Parser/CircuitBuilder/CircuitBuilderUtils.cpp \
	Parser/FloatExpressionEvaluator.cpp \
	CircuitPrinter.cpp \
	Matrix.cpp \
	Simulator.cpp \
	main.cpp

$(ODIR)/Parser/ASTGenerator.o: CXXFLAGS:=$(filter-out $(OPTIFLAGS),$(CXXFLAGS))

# Maccro that contains the default include repository with all *.hpp headers.
INC= -Iinclude -I/usr/include/boost

# Maccro that contains all the entries of CUDA's headers.
CUINC= -Icuda_include -I$(CUDA_HOME)/samples/common/inc

# Maccro that combines both includes.
BOTHINC= $(INC) $(CUINC)

# Below we have different types of flags related to NVCC compiler.

# 'NVCCFLAGS' and 'NVCCFLAGS_LINK' are the flags that will be used
# when the architecture of the project doesn't involve any kind of
# dependecies resolution. I.e. when "device" codes are defined
# in files where they are initially called.
NVCCFLAGS= -std=c++14 -arch=sm_61

NVCCFLAGS_LINK= -std=c++14 -arch=sm_61

# 'NVCCFLAGS_COMPILE' and 'NVCCFLAGS_DLINK' have the opposite behaviour
# of the two maccros above. Indeed, those Maccros will be used when the
# architecture of the project involves definition of 'device' codes outside
# of files where they are initially called. Therefore to avoid any file with
# a lot of lines of codes, we will mostly use those maccros.
# LINK: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#using-separate-compilation-in-cuda
NVCCFLAGS_COMPILE= -std=c++14 -arch=sm_61 --device-c

NVCCFLAGS_DLINK= -std=c++14 -arch=sm_61 --device-link

# This maccro is related to 'NVCCFLAGS_COMPILE' and 'NVCCFLAGS_DLINK' maccros.
# Indeed,this maccro contains the name of the object that will be built thanks
# to those maccros (dlink.o), and where it will be stored (see 'CUODIR' maccro).
DLINKOBJ= $(CUODIR)/dlink.o

# objects from .cpp source files.
_OBJS=	$(SRC:.cpp=.o)
OBJS=	$(patsubst %,$(ODIR)/%,$(_OBJS))

# objects from .cu source files.
_CUOBJS= $(CUSRC:.cu=.o)
CUOBJS=	$(patsubst %,$(CUODIR)/%,$(_CUOBJS))

# Maccro that contains all the dynamic libraries that will be linked
# to the created objects in order to get our executable.
LDIR=	-L$(CUDA_HOME)/lib64 -lcuda -lcudart -lboost_system -lboost_filesystem

# Default rule that will be called when the user types 'make'.
# Here, it will create 'ODIR' and 'CUODIR' repositories and call the 'NAME' rule.
all:	$(ODIR) $(CUODIR) $(NAME)

# Rule that creates the 'ODIR' directory.
$(ODIR):
	mkdir $(ODIR)
	mkdir $(ODIR)/Parser
	mkdir $(ODIR)/Parser/CircuitBuilder

# Rule that creates for each *.cpp file a *.o file,
# with the specified compilation line (see line 103).
$(ODIR)/%.o:	$(SDIR)/%.cpp
		$(CXX) $(CXXFLAGS) $(BOTHINC) -o $@ -c $<

# Rule that creates the 'CUODIR' directory.
$(CUODIR):
	mkdir $(CUODIR)

# Rule that creates for each *.cu file a *.o file,
# with the specified compilation line (see line 112).
$(CUODIR)/%.o:	$(CUSDIR)/%.cu
		$(NVCC) $(BOTHINC) $(NVCCFLAGS_COMPILE) -o $@ -c $<

# Main rule that performs the compilation of each source file (see line 116),
# and links all the compiled objects with specific links (see lines 117 and 118).
$(NAME):	$(OBJS) $(CUOBJS)
		$(NVCC) $(NVCCFLAGS_DLINK) -o $(DLINKOBJ) $(CUOBJS) $(LDIR)
		$(CXX) -o $(NAME) $(OBJS) $(CUOBJS) $(DLINKOBJ) $(LDIR)

# Rule that deletes the 'ODIR' and 'CUODIR' repositories.
clean:
	$(RM) $(ODIR) $(CUODIR)

# Rule that calls the 'clean' rule, and deletes the created executable.
fclean:	clean
	$(RM) $(NAME)

# Rule that respectively calls the 'fclean' and 'all' rules.
re:	fclean all

# Specific rule that checks the Makefile's cycle by testing all the specified rules.
.PHONY: all clean fclean re
