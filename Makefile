CXX=	g++

NVCC=	nvcc

RM=	rm -rf

NAME=	simplePrintf

# .cpp source files directory
SDIR= src

# .cu source files firectory
CUSDIR= cuda_src

# objects from .cpp sources directory
ODIR=	obj

# objects from .cu sources directory
CUODIR= cuda_obj

# .cu source files
CUSRC=	simplePrintf.cu

# .cpp sources
SRC=	main.cpp

# Includes for CXX
INC= -Iinclude -Icuda_include

# Includes for NVCC
CUINC= -I $(CUDA_HOME)/samples/common/inc/ -Icuda_include

# objects from .cpp source files
_OBJS=	$(SRC:.cpp=.o)
OBJS=	$(patsubst %,$(ODIR)/%,$(_OBJS))

# objects from .cu source files
_CUOBJS=	$(CUSRC:.cu=.o)
CUOBJS=	$(patsubst %,$(CUODIR)/%,$(_CUOBJS))

LDIR=	-L$(CUDA_HOME)/lib64 -lcuda -lcudart

all:	$(ODIR) $(CUODIR) $(NAME)

# create objects from .cpp source files directory
$(ODIR):
	mkdir $(ODIR)

# create objects from .cpp source files
$(ODIR)/%.o:	$(SDIR)/%.cpp
		$(CXX) $(INC) -o $@ -c $<

# create objects from .cu source files directory
$(CUODIR):
	mkdir $(CUODIR)

# create objects from .cu source files
$(CUODIR)/%.o:	$(CUSDIR)/%.cu
		$(NVCC) $(CUINC) -o $@ -c $<

$(NAME):	$(OBJS) $(CUOBJS)
		$(CXX) -o $(NAME) $(OBJS) $(CUOBJS) $(LDIR)

clean:
	$(RM) $(ODIR) $(CUODIR)

fclean:	clean
	$(RM) $(NAME)

re:	fclean all

.PHONY: all clean fclean re
