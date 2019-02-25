This project implements parallel kNN in java.

Instalation:
	mpj is required to run the project. It has been tested and works with mpj 0.44, a copy of which can be found in thrirdParty/mpj-v0_44 along 
	with details on how to install for your operating system.

	The project has been tested and works with java 1.8. You will need the SDK for this version of java or latter.

Compiling:
	The project can be compiled using the supplied eclipse project. It can be compiled using another ide or the command line, 
	however, the mpj and jomp jar files will need to be added to the classpath used when compiling.

Running:
	See the mpj readme in thrirdParty/mpj-v0_44 for information on running mpj programs. To run using 4 processes on one computer on windows,
	open command prompt and change directory to the bin directory then execute mpjrun -np 4 Main.