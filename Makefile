.PHONY: dirs downloads all clean

ifdef OS
   RM = rmdir /s /q
else
   ifeq ($(shell uname), Linux)
      RM = rm -rf
   endif
endif

dirs:
	@mkdir "./data/"

	@mkdir "./results/"

	@echo Data directories created.

downloads:
	wget -O data/concrete_compressive_strength.zip "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip"

unzip:
	unzip data/concrete_compressive_strength.zip -d data/

all: dirs downloads
	echo All done.

clean:
	@$(RM) "./data/"
	@$(RM) "./results/"
	@echo Clean done.