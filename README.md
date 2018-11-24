# ccia-kmode
Cluster Center Initialization Algorithm for K-mode Clustering

This distribution contains three java files:

1. Kmode.java -- This is the main class file to generate initial modes, generate cluster strings, perform K-modes clustering and hierarchical clustering, along with other related functions

2. initKmodeProm.java -- This is the test file that uses Kmode.java to generate initial modes and perform K-modes clustering afterwards

3. testKmode.java -- This is the test file to perform normal K-modes clustering with random initial modes

There are two things that needs to be set before executing the initiKmodeProm.java or testKmode.java
- The path of source file name. The file name should be in arff format

  String targetDir = "//home/shehroz//workspace//Clustering//data//";  //Directory name

String inputFile =  targetDir+"soybean-small.arff"; //input file name
		
- The number of clusters in the data. It can be done by altering this line 

  initkm.setK(N);//number of clusters

  where N is the number of clusters in the data

If eclipse is not used then the following line can be removed from the top

package initCategorical;

If you use this code for your research publication, please cite the following paper

@article{khan2013cluster,
  title={Cluster center initialization algorithm for K-modes clustering},
  author={Khan, Shehroz S and Ahmad, Amir},
  journal={Expert Systems with Applications},
  volume={40},
  number={18},
  pages={7444--7456},
  year={2013},
  publisher={Elsevier}
}
