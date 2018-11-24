//This is the main algorithm that uses prominent attributes and/or all attributes if needed
// merge them
//Author: Shehroz S. Khan
//Affiliation: University of Waterloo, Canada
//Date: May'2012
//LICENCE: Read Separate File

//About: This is the main class to generate initial modes and perform K-modes clustering
//       Details of the algorithm are in the paper

package initCategorical;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Map.Entry;

import weka.clusterers.HierarchicalClusterer;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class Kmode {

	private int K;
	private int ITR_MAX;
	private int [] count; 
	private String [][] newModes;
	private int [][] mCluster ;
	
	//Constructor
	public Kmode () {
	}

	//Setters 
	public void setK(int k) {
		K = k;
	}

	public void setITR_MAX (int itr){
		ITR_MAX=itr;
	}
	
	//Getters
	public int getK() {
		return K;
	}

	public String [][] getModes(){
		return newModes;
	}
	
	public int [][] getmCluster() {
		return mCluster;
	}
	
	public int [] getObjectCountInClusters(){
		return count;
	}

	public int getITR_MAX (){
		return ITR_MAX;
	}

	//Create random initial cluster centers
	public String [][] randomInitializeCenter(Instances data){

		Random rand = new Random();
		data.randomize(rand); //randomize data
		int dataInK = data.numInstances()/K; //adjust number of instances in each K
		int remDataInK = data.numInstances() - dataInK*(K-1); // numbers in last cluster
		System.out.println("d="+dataInK+" rd="+remDataInK);
		int init=0;
		String [][] modes = new String [K][data.numAttributes()];
		for(int i=0;i<K-1;i++){ //till K-1
			int j = 0;
			for(int k=0;k<data.numAttributes();k++){
				String [] val = new String [dataInK];
				for(j=init;j<dataInK+init;j++){
					val[j-init] = data.instance(j).stringValue(k); //store all attribute values
					//System.out.print("*"+val[j-init]+" ");
				}
				//System.out.println();
				modes[i][k]=computeMode(val);
			} //end for k
			/*
			for(int k=0;k<data.numAttributes();k++)
				System.out.print(modes[i][k]+" ");
			System.out.println();
			 */
			init=j;
			//System.out.println("init="+init);
		} //end for i

		//Compute modes for last cluster
		//System.out.println("Last cluster, init="+init);
		for(int k=0;k<data.numAttributes();k++){
			String [] val = new String [remDataInK];
			for(int j=init;j<init+remDataInK;j++) {
				val[j-init] = data.instance(j).stringValue(k);
				//System.out.print("#"+val[j-init]+" ");
			}
			//System.out.println();
			modes[K-1][k]=computeMode(val);
		}
		return modes;
	}//end randomInitializaCenter

	// Kmode Clustering
	public int[] clustering (Instances data, String [][] modes, int K){
		int membership [] = new int [data.numInstances()];
		for(int itr=0;itr<ITR_MAX;itr++) {
			count = new int [K];
			mCluster = new int [K][data.numInstances()];
			
			System.out.println("------------------ITR="+(itr+1)+"-----------------------");
			//Partition data based on initial modes
			for(int i=0;i<data.numInstances();i++){
				int distance [] = new int [K];
				for(int j=0;j<K;j++){
					String [] str = new String [data.numAttributes()]; 
					for(int k=0;k<data.numAttributes();k++)
						str[k] = data.instance(i).stringValue(k);
					distance[j] = computeHammingDistance(str,modes[j], data.numAttributes());
				} //end for j
				//Find membership of instances to clusters w.r.t. hamming distance
				//for(int j=0;j<K;j++)
					//System.out.print(distance[j]+" ");
				membership[i] = findClusterMembership(distance);
				//System.out.println("m="+membership[i]+" c="+count[membership[i]]);
				mCluster[membership[i]][count[membership[i]]]=i;
				count[membership[i]]++;
				//System.out.println("*m="+membership[i]);
			} //end for i
			for(int i=0;i<K;i++)
				System.out.print("cluster["+i+"]="+count[i]+" ");
			System.out.println();
			//Allocate instances to K clusters
			newModes = new String [K][data.numAttributes()];
			for(int i=0;i<K;i++){
				for(int k=0;k<data.numAttributes();k++) {
					String [] val = new String [count[i]];
					for(int j=0;j<count[i];j++){
						//System.out.print(data.instance(mCluster[i][j]).stringValue(k)+" ");
						val[j] = data.instance(mCluster[i][j]).stringValue(k);
					} //end for j
					newModes[i][k]= computeMode(val);
					//System.out.print(newModes[i][k]+" ");
				} //end for k
				//System.out.println();
			} //end for i
			
			//Check termination condition
			//System.out.println("D="+data.numAttributes());
			int flag=1;
			for(int i=0;i<K;i++){
				for(int j=0;j<data.numAttributes();j++){
					if(modes[i][j].equals(newModes[i][j])) flag*=1;
					else flag=0;
				}
			}
			if(flag==0) modes=newModes;
			else if (flag==1) break;
		} //end for itr


		return membership;

	}

	public int findClusterMembership(int[] distance) {
		int [] temp = new int [distance.length];
		for(int i=0;i<temp.length;i++) temp[i]=distance[i];
		Arrays.sort(temp);
		int i;
		for(i=0;i<distance.length;i++){
			if(temp[temp.length-1]==distance[i])
				break;
		}
		return i;
	}

	// Compute Hamming Distance
	public int computeHammingDistance(String[] str, String[] strings, int attributes) {

		int dist=0;
		for(int i=0;i<attributes;i++) {
			if(str[i].equals(strings[i]))
				dist++;
		}
		return dist;
	}

	//Compute mode
	public String computeMode (String [] args){
		Map<String, Integer> m = new HashMap<String, Integer>();
		for (String a : args) {
			Integer freq = m.get(a);
			m.put(a, (freq == null) ? 1 : freq + 1);
		}
		//System.out.println(m.size() + " distinct words:" + m);

		LinkedHashMap ms = sortByValue(m);
		List<Entry<String,Integer>> entryList = new ArrayList<Map.Entry<String, Integer>>(ms.entrySet());
		Entry<String, Integer> lastEntry = entryList.get(entryList.size()-1);
		//System.out.println("#"+lastEntry+" --"+lastEntry.getKey().toString());
		return lastEntry.getKey().toString();
	} //end computeMode

	//Sort a map by value
	public LinkedHashMap sortByValue(Map<String, Integer> map) {
		List list = new LinkedList(map.entrySet());
		Collections.sort(list, new Comparator() {
			public int compare(Object o1, Object o2) {
				return ((Comparable) ((Map.Entry) (o1)).getValue())
				.compareTo(((Map.Entry) (o2)).getValue());
			}
		});

		Map result = new LinkedHashMap();
		for (Iterator it = list.iterator(); it.hasNext();) {
			Map.Entry entry = (Map.Entry)it.next();
			result.put(entry.getKey(), entry.getValue());
		}
		return (LinkedHashMap) result;
	} //end sortByValue
	
	//Finding distinct classes in the data
	public String [] findDistinctClasses(Instances data){
		String [] distinctClasses = new String [data.numDistinctValues(data.classIndex())]; 
		for(int i=0;i<data.numDistinctValues(data.classIndex());i++) {
			distinctClasses[i] = data.classAttribute().value(i);
			//System.out.print("d["+i+"]="+distinctClasses[i]+" ");
		}
		return distinctClasses;
	}
			
	//find actual modes of the data
	public String[][] findActualModes(Instances data){
		//System.out.println();
		String [] distinctClasses = findDistinctClasses(data);
		String [][] actualModes = new String [distinctClasses.length][data.numAttributes()-1];
		//For every class
		for(int i=0;i<distinctClasses.length;i++) {
			//collect all the instances corresponding to a class 
			Instances tempData = new Instances(data,data.numInstances()); //to hold temporary data in between data partitions 
			for(int j=0;j<data.numInstances();j++)
			if(data.instance(j).stringValue(data.classIndex()).equals(distinctClasses[i]))
				tempData.add(data.instance(j));//store them in a temp data
			//Find mode of every attribute
			for(int l=0;l<tempData.numAttributes()-1;l++) {
				String [] val = new String [tempData.numInstances()];
				for(int m=0;m<tempData.numInstances();m++) {
					val[m]=tempData.instance(m).stringValue(l);
				}
				actualModes[i][l]=computeMode(val);
			}
		}
		return actualModes;
	}

	//Find initial modes using prominent attributes
	public String[][] findInitialModes(int numK, Instances data, String outputFile) throws Exception {
		//Step-1 : Find Prominent Attributes
		//In every attribute
		ArrayList<Integer> promAttribute = new ArrayList<Integer>(); //to store prominent attributes
		
		for(int j=0;j<data.numAttributes();j++) {
			String [] tempAtt =  new String [data.numInstances()];
			//Find distinct number of attribute values
			for(int i=0;i<data.numInstances();i++) {
				//System.out.print(data.instance(i).stringValue(j)+" ");
				tempAtt[i] = data.instance(i).stringValue(j);
				}
			//System.out.println();
			Map<String, Integer> distinctAtt = distinctAttributes(tempAtt);
			//System.out.println(" --> distinctAtt="+distinctAtt+ " size="+distinctAtt.size());
			// if distinct values are less than numK and greater than one, then add it as PRominent Attribute
			if(distinctAtt.size()<=numK && distinctAtt.size()>1){
				promAttribute.add(j);
				//countPromAttrib++;
				}
			} //end for j
		//promAttribute.clear(); //Shortcut To choose all attributes
		//Step 2 - if there are no Prominent Attributes, then consider all attributes (as Prominent Attributes)
		if(promAttribute.size()==0) {
			for(int i=0;i<data.numAttributes();i++)
				promAttribute.add(i);
			}
			
		
		//Vanilla - Consider all attributes
		/*for(int i=0;i<data.numAttributes();i++)
			promAttribute.add(i);
		*/
		System.out.println("Prom Attrib="+promAttribute.size());
		System.out.print("Attributes to consider for initialization ");
		/*if(promAttribute.size()>0)
			System.out.print("(Step1): ");
		else
			System.out.print("(Step2):");*/
		for(int i=0;i<promAttribute.size();i++)
			System.out.print(promAttribute.get(i)+" ");
		System.out.println();
		
		//Generate Class Strings
		String [] cstr=generateClusterStrings(promAttribute, data, outputFile);
		//Get Initial Modes from these strings
		String[][] initialModes = generateModesHierarchical(data, cstr, outputFile, numK);
		
		return initialModes;
		} //end for findNewModes()
	
	// Compute Distinct Attribute Values
	public Map<String, Integer> distinctAttributes (String [] args){
		Map<String, Integer> m = new HashMap<String, Integer>();
		for (String a : args) {
			Integer freq = m.get(a);
			m.put(a, (freq == null) ? 1 : freq + 1);
			}
		//System.out.println(m.size() + " distinct words:" + m);
		return m;
	}

	public String[] generateClusterStrings(ArrayList<Integer>promAttribute, Instances data, String outputFile) throws IOException {
	    //For every 'selected' attribute find the number of distinct values
		int [][] clusterString = new int [data.numInstances()][data.numAttributes()];
		for(int j=0;j<promAttribute.size();j++) {
			System.out.println("\n\t<<<Attribute "+promAttribute.get(j)+">>>");
			String [] tempAtt =  new String [data.numInstances()];
			for(int i=0;i<data.numInstances();i++) {
				//System.out.print(data.instance(i).stringValue(promAttribute.get(j))+" ");
				tempAtt[i] = data.instance(i).stringValue(promAttribute.get(j));
				}
			Map<String, Integer> distinctAtt = distinctAttributes(tempAtt);
			//System.out.println(" --> distinctAtt="+distinctAtt);
			
			//for every distinct value, partitions the data in number of clusters equal to distinct attributes
			int tcount=0;
			String [][] intermodes = new String [distinctAtt.size()][data.numAttributes()];
			for(Iterator<String> it = distinctAtt.keySet().iterator();it.hasNext();){
				String key = it.next().toString();///distinct attrib value
				//System.out.print("*"+key+" ");
				Instances tempData = new Instances(data,data.numInstances()); //to hold temporary data in between data partitions 
				//create a temp data that corresponds to all data containing the 'key' attrib value
				for(int k=0;k<data.numInstances();k++){
					if(key.equals(data.instance(k).stringValue(promAttribute.get(j)))) {
						tempData.add(data.instance(k));
						}
					} //end for k
				//System.out.println(tempData);
				//Find mode of every attribute
				//System.out.println("m="+tempData.numInstances());
				for(int l=0;l<tempData.numAttributes();l++) {
					String [] val = new String [tempData.numInstances()];
					for(int m=0;m<tempData.numInstances();m++) {
						val[m]=tempData.instance(m).stringValue(l);
					}
					intermodes[tcount][l]=computeMode(val);
				}
				//System.out.println();
				tcount++;
				} //end for iterator
			/*for(int l=0;l<tcount;l++) {
				for(int m=0;m<data.numAttributes();m++){
					System.out.print("*"+intermodes[l][m]+" ");
					}
				System.out.println();
				}
				*/
			int [] cs = new  int[data.numInstances()];
			
			cs=clustering(data, intermodes,distinctAtt.size());//call Kmode with new modes on the original data
			//Storing class string
			//System.out.println("L="+cs.length);
			for(int l=0;l<data.numInstances();l++) {
				//System.out.print(cs[l]+" ");
				clusterString[l][j]=cs[l];
			}
			//System.out.println();
			
		} //end for j
		
		//Convert numeric class string to character strings
		String [] cstr = new String [data.numInstances()];
		for(int i=0;i<data.numInstances();i++) {
			cstr[i]="";
			for(int j=0; j<promAttribute.size()-1;j++){
				cstr[i]+=clusterString[i][j]+",";
				//System.out.print(clusterString[i][j]+", ");
			}
			cstr[i]+=clusterString[i][promAttribute.size()-1];
			//System.out.println(cstr[i]);
			//System.out.println();
		}
		
		//Find distinct class strings
		Map<String, Integer> distinctClassStr = distinctAttributes(cstr);
		//System.out.println(distinctClassStr);
		//Sort them
		LinkedHashMap sortedCS = new LinkedHashMap();
		sortedCS = sortByValue(distinctClassStr);
		//System.out.println("$"+sortedCS);
		System.out.println("\nDistinct Cluster Strings="+distinctClassStr.size());
		
		//if distinct class strings are greater than sqrt(N) then choose the top sqrt(N) class strings
		int limit = (int) Math.ceil(Math.sqrt(data.numInstances()));
		ArrayList<String> topclusterString = new ArrayList<String>();
		System.out.println("Limit=SQRT(N)="+limit);
		if(distinctClassStr.size() > limit) {
			int topC=0;
			int i=0;
			Iterator it = sortedCS.entrySet().iterator();
			while (it.hasNext()) {
				Map.Entry pairs = (Map.Entry)it.next();
				//System.out.println("i="+i+" s="+(sortedCS.size()-limit)+" "+pairs.getKey() + " = " + pairs.getValue());
				//Store last 'limit' strings
				if(i>=(sortedCS.size()-limit)) {
					//topclusterString[topC]=pairs.getKey().toString();
					topclusterString.add(topC, pairs.getKey().toString());
					topC++;
					}
				i++;
				}
			} //end for if
		// or else use the present strings as is
		else if(distinctClassStr.size() <= limit) {
			int i=0;
			Iterator it = sortedCS.entrySet().iterator();
			while (it.hasNext()) {
				Map.Entry pairs = (Map.Entry)it.next();
				//System.out.println("i="+i+" s="+(sortedCS.size()-limit)+" "+pairs.getKey() + " = " + pairs.getValue());
				//Store all strings
				//topclusterString[i]=pairs.getKey().toString();
				topclusterString.add(i, pairs.getKey().toString());
				i++;
				}
			}//end for else 
		
		//Write class strings in output.csv
		BufferedWriter output = new BufferedWriter(new FileWriter(outputFile));
		//output.write("prom="+promAttribute.size()+" "+promAttribute+"\nsize="+distinctClassStr.size()+" limit="+limit);
		//output.newLine();
		System.out.println("Chosen Cluster Strings:");
		for(int i=0;i<topclusterString.size();i++)
			System.out.println("^"+topclusterString.get(i));
		for(int i=0;i<promAttribute.size()-1;i++)
			output.write("a"+i+",");
		output.write("a"+(promAttribute.size()-1)+"\n");
		for(int i=0;i<topclusterString.size();i++){
			output.write(topclusterString.get(i));
			output.newLine();
			}
		output.close();
		
		return cstr;
		}//end for generateClusterStrings()

	//Use hierarchical clustering to merge similar cluster strings into K clusters
	public String[][] generateModesHierarchical(Instances data, String [] cstr, String outputfile, int K) throws Exception{
		DataSource source = new DataSource(outputfile);
		Instances hdata = source.getDataSet();
		//System.out.println(hdata);
		
		//Apply hierarchical clustering on these class strings
		HierarchicalClusterer hcl = new HierarchicalClusterer();
		SelectedTag linkType = new SelectedTag("SINGLE", HierarchicalClusterer.TAGS_LINK_TYPE);
		hcl.setNumClusters(K);
		hcl.setLinkType(linkType);
		System.out.println(">>>Type="+linkType.getSelectedTag());
		hcl.buildClusterer(hdata);
		int classLabels [] = new int [hdata.numInstances()];
		System.out.println("Merging of cluster strings in respective clusters:");
		for(int i=0;i<hdata.numInstances();i++) {
			classLabels[i]=hcl.clusterInstance(hdata.instance(i));
			System.out.print(classLabels[i]+" ");
		}
		System.out.println();
		//From the class strings that are clustered in same clusters, find the corresponding patterns
	    String [][] initialModes = new String [K][data.numAttributes()];
	    //int [] flag = new int [data.numInstances()];
	    for(int i=0;i<K;i++){ //for every cluster
	    	Instances tempData = new Instances(data, data.numInstances()); //to hold temp data
	    	for(int j=0;j<hdata.numInstances();j++) { //if reduced class string 
	    		if(classLabels[j]==i) { //falls in ith cluster
	    			//System.out.println("cl="+classLabels[j]);
	    			//then find corresponding patterns in original class string which equals to this reduced class string 
	    			for(int k=0;k<data.numInstances();k++) { 
	    				//System.out.println("hdata="+hdata.instance(j).toString()+" cstr="+cstr[k]);
	    				if(hdata.instance(j).toString().equals(cstr[k])) { 
	    					//System.out.println(" *"+k+" ");
	    					tempData.add(data.instance(k)); //allocate that pattern in this cluster
	    					//flag[k]=1; //mark it as traversed
	    					}
	    				} //end for k
	    			//System.out.println();
	    			} //end if
	    		} //end for j
	    	//System.out.println("***"+tempData);
	    	for(int l=0;l<tempData.numAttributes();l++) {
	    		String [] val = new String [tempData.numInstances()];
	    		for(int m=0;m<tempData.numInstances();m++) {
	    			val[m]=tempData.instance(m).stringValue(l);
	    			}
	    		initialModes[i][l]=computeMode(val);
	    		}
	    	} //end for i
	    /*
	    //Printing initial K-Modes
	    for(int i=0;i<K;i++){
	    	for(int j=0;j<data.numAttributes();j++)
	    		System.out.print(initialModes[i][j]+" ");
	    	System.out.println();
	    	}
	    */
	    return initialModes;
	    }		

	public Instances readInputFile(String inputCSVfile) throws Exception {
		
		DataSource source = new DataSource(inputCSVfile);
		Instances data = source.getDataSet();
		// setting class attribute if the data format does not provide this information
		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		//System.out.println(data);

		return data;
		} //end readInputFile
	
	public Instances removeClassAttribute(Instances data) throws Exception {
		// generate data for clusterer (w/o class)
		Remove filter = new Remove();
		filter.setAttributeIndices("" + (data.classIndex() + 1));
		filter.setInputFormat(data);
		Instances dataClusterer = Filter.useFilter(data, filter);
		
		return dataClusterer;
		} //end for removeClassAttribute()
	
	//Match Metric between initial and actual modes
	public double distInitialActual(String [][] initialModes, String [][] actualModes, int K, int M) {
		double match =0;
		for(int i=0;i<K;i++) {
			for(int j=0;j<M;j++) {
				if(initialModes[i][j].equals(actualModes[i][j])) 
					match++;
			}
		}

		return (match/(K*M));
	}
	
	public void clusterToClassEvaluation(Instances data, Kmode initkm, Instances clusterdata) {
		String [] distinctClasses = findDistinctClasses(data);
		String classLabel [];
		int[] countK = initkm.getObjectCountInClusters();
		for(int i=0;i<initkm.getK();i++){
			System.out.println("objects in cluster "+i+"="+countK[i]);
		}
		//System.out.println("C="+data.numDistinctValues(data.classIndex()));
		int classDistribution [][] = new int [data.numClasses()][data.numClasses()];
		String [] clusterLabel = new String [data.numClasses()];
		for(int i=0;i<initkm.getK();i++){
			classLabel = new String[countK[i]];
			for(int j=0;j<countK[i];j++) {
				//System.out.print(initkm.getmCluster()[i][j]+" "+clusterdata.instance(initkm.getmCluster()[i][j]));
				for(int k=0;k<data.numInstances();k++){
					String tempC=new String();
					String tempD=new String();
					for(int l=0;l<clusterdata.numAttributes();l++) {
						tempD+=data.instance(k).stringValue(l);
						tempC+=clusterdata.instance(initkm.getmCluster()[i][j]).stringValue(l);
						}
					if(tempC.equals(tempD)){
						//System.out.println(" -->"+ data.instance(k).stringValue(data.classIndex()));
						classLabel[j]=data.instance(k).stringValue(data.classIndex());
					}
				} //end for k;
				for(int l=0;l<data.numDistinctValues(data.classIndex());l++) {
					if (classLabel[j].equals(distinctClasses[l])) {
						classDistribution[i][l]++;
					}
				}
			} //end for j
			clusterLabel[i]=computeMode(classLabel);//Hopefully this is the cluster label;but not true if misclassification is high
			//System.out.println();
		} //end for i
		System.out.println("Cluster-Class Distribution:-");	
		for(int i=0;i<data.numClasses();i++)
			System.out.print("\t"+distinctClasses[i]);
		System.out.println();
		for(int i=0;i<data.numClasses();i++){
			System.out.print(clusterLabel[i]+"\t");
			for(int j=0;j<data.numClasses();j++) 
				System.out.print(classDistribution[i][j]+"\t");
			System.out.println();
			}

	}
}