package massive.clearsight.Vote_Rate_Entropy;


import java.io.File;
import java.util.List;
import java.util.Set;

import massive.clearsight.Vote_Rate_Entropy.Reviews.distType;
import massive.clearsight.Vote_Rate_Entropy.Reviews.orderType;
import massive.clearsight.Vote_Rate_Entropy.Reviews.weightType;
import massive.clearsight.linguistic_resources.ResourceDefinitions;
import massive.clearsight.linguistic_resources.ResourceLoader;
import massive.clearsight.utils.FileLoader;
import massive.clearsight.utils.ReadWriteFile;

public class TestSelected {

	
//	public static Set<String> stopList = null; 
//	private static Set<String> externalNercWords = null; 
	private static String language = ResourceDefinitions.LANGUAGE_EN;



	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		
		// "C:\UniDatiSperimentali\PRODUCT REVIEW RUGGIERO\VOTE-RATING-ENTROPY\Dataset\Electronics_5.json" "C:\UniDatiSperimentali\PRODUCT REVIEW RUGGIERO\VOTE-RATING-ENTROPY\OutNew\Out.Electronics_5.json.textOUT.VOCAB.txt" 
		// "C:\UniDatiSperimentali\PRODUCT REVIEW RUGGIERO\VOTE-RATING-ENTROPY\OutNew\Useful.Out.Electronics_5.json.txt" "C:\UniDatiSperimentali\PRODUCT REVIEW RUGGIERO\VOTE-RATING-ENTROPY\OutNew\Out" by_NumRev_prod 20 electronicsEntropy 1 3 0 100000 1000000 5
		
		
		String DATA_PATH = args[0];						// !!! fare Pc category ! Location data reviews records  "C:\UniDatiSperimentali\PRODUCT REVIEW RUGGIERO\VOTE-RATING-ENTROPY\REVIEWS_MOBILE.csv"
		String VOCAB_PATH = args[1];					// Location vocabulary	"C:\UniDatiSperimentali\PRODUCT REVIEW RUGGIERO\proveDLWord2Vec\Mobiles\files\REVIEWS.MobileOUT.NotNormal.VOCAB.txt"
		String USEFULWORDS_PATH = args[2];				// Location info useful	 "C:\UniDatiSperimentali\PRODUCT REVIEW RUGGIERO\VOTE-RATING-ENTROPY\Useful.txt"  the features
		String OUT_PATH = args[3];						// output path "C:\UniDatiSperimentali\PRODUCT REVIEW RUGGIERO\VOTE-RATING-ENTROPY\\OutNew\\Out"
		String typeList = args[4];	 					/*	Ex.:			switch (typeList) {  // listType {by_lengh, by_prodID, by_NumRev_prod};		Integer			
																 case by_lengh:  if (r.getTextLength() > value) this.add(r);
																 break;
																 case by_prodID:  if (r.getProdID().equal(value)) this.add(r);  String because in some case we use asim code amazon  4384
																 break;
																 case by_NumRev_prod:  if (numReviewsForProd.get(r.getProdID()) > value) this.add(r);	Integer		
																 break;
																 case by_votesUp:  if (r.getHelpfulVote() > Integer.parseInt(value)) this.add(r);
				 												 break;

																}
														*/
		String value =  args[5];						// value of typeList  
		String category = args[6];
		int typeParser =  Integer.parseInt(args[7]);	// 0 = traditional parser format DB Massive, 1 = parser format data http://jmcauley.ucsd.edu/data/amazon/
		int minTextLen =  Integer.parseInt(args[8]);	// int minTextLen (# words) = min Threshold reviews
		int fromLine =  Integer.parseInt(args[9]);		// from Lines loaded
		int toLine =  Integer.parseInt(args[10]);		// to Lines loaded
		int nDocs   =  Integer.parseInt(args[11]);		// num docs of category Pc = 400K
		int window =  Integer.parseInt(args[12]);		// window to take context of current review

		

		try {
			
			
			// INITIALIZATION
			String PathStopList = ResourceDefinitions.STOPWORDS_RESOURCE_NAME;	
			Reviews.stopList = FileLoader.loadLines( ResourceLoader.getResourceAsStream(PathStopList,Reviews.language) );
			// load nerc			
			String externalNercFilename = ResourceDefinitions.ENTITY_RESOURCE_NAME;	
			Reviews.externalNercWords = FileLoader.loadLines(ResourceLoader.getResourceAsStream(externalNercFilename,language));
			System.out.println("LOADING entity-names pre-defined GENERAL...externalNercWords size "+Reviews.externalNercWords.size());
			System.out.println("Category: "+category);
			externalNercFilename = ResourceDefinitions.ENTITY_RESOURCE_NAME2;	
			Reviews.externalNercWords.addAll(FileLoader.loadLines(ResourceLoader.getResourceAsStream(externalNercFilename,language)));
			System.out.println("LOADING nerc-pmi-general ... externalNercWords size "+Reviews.externalNercWords.size());
			Reviews.externalNercWords.addAll(FileLoader.loadLines(ResourceLoader.getResourceAsStream(category+ "/nerc-pmi",language)));
			System.out.println("LOADING nerc-pmi-local ... externalNercWords size "+Reviews.externalNercWords.size());
			Reviews.NERCSpecialized = FileLoader.loadLines(ResourceLoader.getResourceAsStream(category+ "/nerc-pmi",language));
			Reviews.positiveWords = FileLoader.loadLines(ResourceLoader.getResourceAsStream(category+ "/positiveWords",language));
			System.out.println("LOADING positiveWords ... positiveWords size "+Reviews.positiveWords.size());
			Reviews.negativeWords = FileLoader.loadLines(ResourceLoader.getResourceAsStream(category+ "/negativeWords",language));
			System.out.println("LOADING negativeWords ... negativeWords size "+Reviews.negativeWords.size());

			
			Reviews.vocab.loadVocab (VOCAB_PATH);
	        List<String> reviews =  FileLoader.loadSortedLines(DATA_PATH, fromLine, toLine ); 		//!!!!!!  use loadSortedLines(DATA_PATH, from, to); for Big File (> 500000 lines) 
	        System.out.println("dim dataset reviews "+reviews.size());                   	 
	        // load only the selected list by criteria: typeList
	        //Reviews rev = new Reviews(reviews, USEFULWORDS_PATH, Reviews.vocab, Reviews.stopList, Reviews.externalNercWords, Reviews.listType.valueOf(typeList), distType.ChebyshevDistance, value );	
			File myFile = new File(DATA_PATH);
			String myDir = myFile.getParent();
	        Reviews rev = new Reviews(reviews, USEFULWORDS_PATH, Reviews.vocab, Reviews.stopList, Reviews.externalNercWords, 
	        		Reviews.listType.valueOf(typeList), distType.ChebyshevDistance, value, typeParser, minTextLen,nDocs, orderType.ProdDateChain, 
	        		OUT_PATH+"."+myFile.getName()+".text.txt", Reviews.NERCSpecialized, Reviews.negativeWords, Reviews.positiveWords  );	        

	        // evaluate best results and Print it on the log file	        
			rev.computeEntropy (weightType.tfIDF, distType.ChebyshevDistance);
	        rev.trendToHelpfulVoteOrderByDate (orderType.SampEnt);							// X-AXIS sample entropy: Y-axis  HelpfulVotePerCent, HelpfulVote UP
	        rev.trendToHelpfulVoteOrderByDate (orderType.Date);								// X-AXIS date: Y-axis sample entropy, HelpfulVotePerCent, HelpfulVote UP
			rev.computeEntropy (weightType.logTf, distType.ChebyshevDistance);
	        rev.trendToHelpfulVoteOrderByDate (orderType.SampEnt);							// X-AXIS sample entropy: Y-axis  HelpfulVotePerCent, HelpfulVote UP
	        rev.trendToHelpfulVoteOrderByDate (orderType.Date);								// X-AXIS date: Y-axis sample entropy, HelpfulVotePerCent, HelpfulVote UP
			
			// GOAL: X-AXIS entropy -> Y-AXIS HelpfulVotePerCent, HelpfulVote UP, Rating

	        // Compute entropy after setting the weight and print the selected list using different way to calculate entropy	 
	        // and SAVE ON .CSV FILE
//	        rev.printSelectedData (OUT_PATH+((typeList.equals("by_prodID")) ? ".PRODID="+value : ""), weightType.one, distType.ChebyshevDistance);
//	        rev.printSelectedData (OUT_PATH+((typeList.equals("by_prodID")) ? ".PRODID="+value : ""), weightType.logTf, distType.ChebyshevDistance);
//	        rev.printSelectedData (OUT_PATH+((typeList.equals("by_prodID")) ? ".PRODID="+value : ""), weightType.tfIDF, distType.ChebyshevDistance);
	        
	        
	        
			rev.printSelectedData (OUT_PATH+typeList+"."+fromLine+"-"+toLine+"."+myFile.getName(), weightType.logTf, distType.ChebyshevDistance, 
					window, orderType.ProdDateChain);		// orderType.ProdDateChain prod uqual -> order by reviews
			
	
			
	        // In order to discovery the kwds of the top voted reviews -> get the 2 dataset UsefulNew.Electronics.txt and a new dataset with all items of clusters
	        // calculate the entropy only using the biggest, but find the matching with the review text using the 2 files: put in evidence the matching words with #
	        
		} catch (Exception e) {
			e.printStackTrace();
		}



	}

	

	

}
