package initCategorical;

import weka.core.Instances;

//This is the main algorithm that uses prominent attributes and/or all attributes if needed
//Author: Shehroz S. Khan
//Affiliation: University of Waterloo, Canada
//Date: May'2012
//LICENCE: Read Separate File

//About: This is the test class to generate initial modes and perform K-modes clustering
//       Details of the algorithm are in the paper

public class initKmodeProm {
	public static void main(String[] args) throws Exception {
		
        //Read input file
		String targetDir = "//home/shehroz//workspace//Clustering//data//";
		String inputFile =  targetDir+"soybean-small.arff";
		String outputFile = targetDir+"output.csv";
		
		Kmode initkm = new Kmode();
		initkm.setK(4);//number of clusters
		initkm.setITR_MAX(1000);
		
		//Reads the input file and prepare a clustered i.e. remove the class attribute
		Instances data = initkm.readInputFile(inputFile);
		Instances clusterdata = initkm.removeClassAttribute(data);
		//clusterdata.randomize(new Random(10)); //to randomize the data
		
		//New modes obtained using Initialization Algorithm
		//initkm.setModes(clusterdata);
		String[][] initialModes = initkm.findInitialModes(initkm.getK(), clusterdata, outputFile);
		System.out.println("\nComputed Initial Modes:");
		for(int i=0;i<initkm.getK();i++) {
			for(int j=0;j<clusterdata.numAttributes();j++)
				System.out.print("*"+initialModes[i][j]+" ");
			System.out.println();
		}
		
		//Perform clustering using the proposed initial modes
		initkm.clustering(clusterdata, initialModes, initkm.getK());
		String[][] finalModes = initkm.getModes();
		System.out.println("\nFinal Modes after clustering with initial modes:");
		for(int i=0;i<initkm.getK();i++) {
			for(int j=0;j<clusterdata.numAttributes();j++)
				System.out.print("*"+finalModes[i][j]+" ");
			System.out.println();
		}

		//Find actual modes of the data
		String [][] actualModes = initkm.findActualModes(data);
		System.out.println("\nActual Modes:");
		for(int i=0;i<initkm.getK();i++) {
			for(int j=0;j<data.numAttributes()-1;j++)
				System.out.print("#"+actualModes[i][j]+" ");
			System.out.println();
		}

		//Distance between initial and actual modes
		double matchMetric = initkm.distInitialActual(initialModes, actualModes, initkm.getK(), data.numAttributes()-1);
		System.out.println("\nmatchMetric="+matchMetric);
		System.out.println("(This metric may not be true, manual re-arrangements of labels " +
				"is needed in arff file to get exact value of metric)\n");
		
		//Cluster to Class Evaluation
		initkm.clusterToClassEvaluation(data, initkm, clusterdata);
		System.out.println("(The class labels on the left are computed using an adhoc " +
				"logic and may not represent actual labels. Manual intervention may be " +
				"needed to get actual class labels)");
		
	} //end of main

} //end of class