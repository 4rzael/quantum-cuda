# @Author: Julien Vial-Detambel <l3ninj>
# @Date:   2018-06-12T11:58:18+01:00
# @Email:  julien.vial-detambel@epitech.eu
# @Project: CUDA-Based Simulator of Quantum Systems
# @Filename: Makefile
# @Last modified by:   l3ninj
# @Last modified time: 2018-06-28T22:43:12+01:00
# @License: MIT License

CXX?=	g++

OPTIFLAGS=	-g3

CXXFLAGS=	-Wextra -Wall -std=c++14 $(OPTIFLAGS)

NVCC=	nvcc

NVCCGLAGS= -std=c++14 -arch=sm_61

RM=	rm -rf

NAME=	quSim

# .cpp source files directory
SDIR=	src

# .cu source files firectory
CUSDIR=	cuda_src

# objects from .cpp sources directory
ODIR=	obj

# objects from .cu sources directory
CUODIR=	cuda_obj

# .cu source files
CUSRC=	QCUDA.cu \
	QCUDA_operations.cu \
	GPUExecutor.cu \
	CPUExecutor.cu \
	ExecutorManager.cu

# .cpp sources
SRC=	Parser/float_expr_ast.cpp \
	Parser/ASTGenerator.cpp \
	Parser/CircuitBuilder/CXBuilder.cpp \
	Parser/CircuitBuilder/MeasureBuilder.cpp \
	Parser/CircuitBuilder/RegisterDeclarationBuilder.cpp \
	Parser/CircuitBuilder/UBuilder.cpp \
	Parser/CircuitBuilder/UserDefinedGateBuilder.cpp \
	Parser/CircuitBuilder/IncludeBuilder.cpp \
	Parser/CircuitBuilder/ResetBuilder.cpp \
	Parser/CircuitBuilder/BarrierBuilder.cpp \
	Parser/CircuitBuilder/ConditionalStatementBuilder.cpp \
	Parser/CircuitBuilder/CircuitBuilder.cpp \
	Parser/CircuitBuilder/CircuitBuilderUtils.cpp \
	Parser/FloatExpressionEvaluator.cpp \
	CircuitPrinter.cpp \
	Matrix.cpp \
	Simulator.cpp \
	main.cpp

$(ODIR)/Parser/ASTGenerator.o: CXXFLAGS:=$(filter-out $(OPTIFLAGS),$(CXXFLAGS))

# Includes for CXX
INC=	-Iinclude -I/usr/include/boost -Icuda_include

# Includes for NVCC
CUINC=	-I$(CUDA_HOME)/samples/common/inc/ -Icuda_include -Iinclude

# objects from .cpp source files
_OBJS=	$(SRC:.cpp=.o)
OBJS=	$(patsubst %,$(ODIR)/%,$(_OBJS))

# objects from .cu source files
_CUOBJS=	$(CUSRC:.cu=.o)
CUOBJS=	$(patsubst %,$(CUODIR)/%,$(_CUOBJS))

LDIR=	-L$(CUDA_HOME)/lib64 -lcuda -lcudart -lboost_system -lboost_filesystem

all:	$(ODIR) $(CUODIR) $(NAME)

# create objects from .cpp source files directory
$(ODIR):
	mkdir $(ODIR)
	mkdir $(ODIR)/Parser
	mkdir $(ODIR)/Parser/CircuitBuilder

# create objects from .cpp source files
$(ODIR)/%.o:	$(SDIR)/%.cpp
		$(CXX) $(CXXFLAGS) $(INC) $(CUINC) -o $@ -c $<

# create objects from .cu source files directory
$(CUODIR):
	mkdir $(CUODIR)

# create objects from .cu source files
$(CUODIR)/%.o:	$(CUSDIR)/%.cu
		$(NVCC) $(CUINC) $(NVCCFLAGS) -o $@ -c $<

$(NAME):	$(OBJS) $(CUOBJS)
		$(CXX) $(CXXFLAGS) -o $(NAME) $(OBJS) $(CUOBJS) $(LDIR)

clean:
	$(RM) $(ODIR) $(CUODIR)

fclean:	clean
	$(RM) $(NAME)

re:	fclean all

.PHONY: all clean fclean re
