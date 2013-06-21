all: build

build: unzip
	g++ -O3 "./Source Code/UCR_ED.cpp" -o ED.bin
	g++ -O3 "./Source Code/UCR_DTW.cpp" -o DTW.bin
	mv ED.bin DTW.bin Executable
	

unzip: trillion.zip 
	unzip trillion.zip

trillion.zip:
	wget http://www.cs.ucr.edu/~eamonn/trillion.zip

clean:
	rm -rf $$(ls | grep -v Makefile)
	rm -rf ./Source\ Code
