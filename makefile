IDIR =include
SRCDIR=src
ODIR=$(SRCDIR)/obj
BINDIR=bin
UTILDIR=utils
BINARYNAME=mandelbrot
CC=gcc
CFLAGS=-I$(IDIR)
LIBS=-lgmp -lmpfr

DEPS := $(shell find $(IDIR) -name '*.h')
SOURCES := $(shell find $(SRCDIR) -name '*.c')
OBJECTS := $(patsubst $(SRCDIR)/%,$(ODIR)/%,$(SOURCES:%.c=%.o))
_UTILS := $(shell find $(UTILDIR) -name '*.c')
UTILS := $(_UTILS:$(UTILDIR)/%.c=$(BINDIR)/%)

all: $(BINDIR)/$(BINARYNAME) $(UTILS)

$(BINDIR)/$(BINARYNAME): $(OBJECTS)
	$(CC) -o $@ $^ $(LIBS)

$(BINDIR)/%: $(UTILDIR)/%.c $(DEPS)
	$(CC) -o $@ $< $(CFLAGS) $(LIBS)

$(ODIR)/%.o: $(SRCDIR)/%.c $(DEPS)
	mkdir -p src/obj
	$(CC) -c -o $@ $< $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(IDIR)/*~
