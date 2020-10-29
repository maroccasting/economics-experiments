package massive.clearsight.Vote_Rate_Entropy;

import java.util.List;

import massive.clearsight.linguistic_resources.ResourceDefinitions;
import massive.clearsight.linguistic_resources.ResourceLoader;
import massive.clearsight.utils.FileLoader;

public class TestGeneral {

	private static String language = ResourceDefinitions.LANGUAGE_EN;


	public static void main(String[] args) {

		
		String DATA_PATH = args[0];						// Location data reviews records  "C:\UniDatiSperimentali\PRODUCT REVIEW RUGGIERO\VOTE-RATING-ENTROPY\REVIEWS_MOBILE.csv"
		String VOCAB_PATH = args[1];					// Location vocabulary	"C:\UniDatiSperimentali\PRODUCT REVIEW RUGGIERO\proveDLWord2Vec\Mobiles\files\REVIEWS.MobileOUT.NotNormal.VOCAB.txt"
		String USEFULWORDS_PATH = args[2];				// Location info useful	 "C:\UniDatiSperimentali\PRODUCT REVIEW RUGGIERO\VOTE-RATING-ENTROPY\Useful.txt"
		String OUT_PATH = args[3];						// output path "C:\UniDatiSperimentali\PRODUCT REVIEW RUGGIERO\VOTE-RATING-ENTROPY\Out.txt"
		int minTextLen =  Integer.parseInt(args[4]);	// int minTextLen = min Threshold reviews
		int nDocs   =  Integer.parseInt(args[5]);		// num docs of category Pc = 400K
		String category = args[6];



		try {
			
			
			// INITIALIZATION
			String PathStopList = ResourceDefinitions.STOPWORDS_RESOURCE_NAME;	
			Reviews.stopList = FileLoader.loadLines( ResourceLoader.getResourceAsStream(PathStopList,Reviews.language) );
			// load nerc
			String externalNercFilename = ResourceDefinitions.ENTITY_RESOURCE_NAME;	
			Reviews.externalNercWords = FileLoader.loadLines(ResourceLoader.getResourceAsStream(externalNercFilename,Reviews.language));
			Reviews.vocab.loadVocab (VOCAB_PATH);
	        List<String> reviews =  FileLoader.loadSortedLines(DATA_PATH);
	        System.out.println("dim dataset reviews "+reviews.size());        
			Reviews.NERCSpecialized = FileLoader.loadLines(ResourceLoader.getResourceAsStream(category+ "/nerc-pmi",language));
	        Reviews rev = new Reviews(reviews, USEFULWORDS_PATH, Reviews.vocab, Reviews.stopList, Reviews.externalNercWords, 0, minTextLen, nDocs, Reviews.NERCSpecialized  );
	        rev.printData (OUT_PATH);

	        
		} catch (Exception e) {
			e.printStackTrace();
		}



	}



	

}
