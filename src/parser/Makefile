CXX=	g++

RM=	rm -rf

NAME=	parser

# .cpp source files directory
SDIR= .

# objects from .cpp sources directory
ODIR=	.

# .cpp sources
SRC=	test_main.cpp

# Includes for CXX
INC= -I/usr/include/boost

# objects from .cpp source files
_OBJS=	$(SRC:.cpp=.o)
OBJS=	$(patsubst %,$(ODIR)/%,$(_OBJS))

all:	$(ODIR) $(NAME)

# create objects from .cpp source files directory
$(ODIR):
	mkdir $(ODIR)

# create objects from .cpp source files
$(ODIR)/%.o:	$(SDIR)/%.cpp
		$(CXX) $(INC) -o $@ -c $<

$(NAME):	$(OBJS)
		$(CXX) -o $(NAME) $(OBJS) $(LDIR)

clean:
	$(RM) $(ODIR)

fclean:	clean
	$(RM) $(NAME)

re:	fclean all

.PHONY: all clean fclean re
