package initCategorical;

import weka.core.Instances;

//Author: Shehroz S. Khan
//Affiliation: University of Waterloo, Canada
//Date: May'2012
//LICENCE: Read Separate File

//About: This is the test class to perform K-modes clustering with random initial modes

public class testKmode {
	public static void main(String[] args) throws Exception {
		
        //Read input file
		String targetDir = "//home/shehroz//workspace//Clustering//data//";
		String inputFile =  targetDir+"soybean-small.arff";
		//String outputFile = targetDir+"output.csv";
		
		//Reads the input file and prepare a clustered i.e. remove the class attribute
		
	    //System.out.println(data);
	    Kmode km1 = new Kmode();
	    km1.setK(4);
		km1.setITR_MAX(1000);
		
		Instances data = km1.readInputFile(inputFile);
		Instances clusterdata = km1.removeClassAttribute(data);
		
		//new String [km1.getK()][data.numAttributes()];
		String [][] initialModes = km1.randomInitializeCenter(clusterdata);
		km1.clustering(clusterdata, initialModes, km1.getK());//perform kmode clustering
		String [][] finalModes=km1.getModes();
		
		System.out.println("\nInitial Modes:");
		for(int i=0;i<km1.getK();i++){
			for(int j=0;j<clusterdata.numAttributes();j++) {
				//output.write(oldModes[i][j]+" ");
				System.out.print(initialModes[i][j]+" ");
			}
			//output.write("\n");
			System.out.println();
		}
		System.out.println("\nFinal Modes:");
		for(int i=0;i<km1.getK();i++){
			for(int j=0;j<clusterdata.numAttributes();j++)
				System.out.print(finalModes[i][j]+" ");
			System.out.println();
		}
		
		String [][] actualModes = km1.findActualModes(data);
		System.out.println("\nActual Modes:");
		for(int i=0;i<km1.getK();i++) {
			for(int j=0;j<clusterdata.numAttributes()-1;j++)
				System.out.print(actualModes[i][j]+" ");
			System.out.println();
		}

		//Distance between initial and actual modes
		double matchMetric = km1.distInitialActual(initialModes, actualModes, km1.getK(), data.numAttributes()-1);
		System.out.println("\nmatchMetric="+matchMetric);
		System.out.println("(This metric may not be true, manual re-arrangements of labels " +
				"is needed in arff file to get exact value of metric)\n");
		
		//Cluster to Class Evaluation
		km1.clusterToClassEvaluation(data, km1, clusterdata);
		System.out.println("(The class labels on the left are computed using an adhoc " +
				"logic and may not represent actual labels. Manual intervention may be " +
				"needed to get actual class labels)");
		} //end of main



} //end of class