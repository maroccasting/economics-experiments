package massive.clearsight.Vote_Rate_Entropy;


import java.io.Serializable;
import java.security.MessageDigest;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.Vector;


import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.deeplearning4j.models.word2vec.VocabWord;

import massive.clearsight.DL4J.tools.ConfigurationSetting;
import massive.clearsight.DL4J.tools.VocabularyFields;
import massive.clearsight.DL4JDemoSentiment.CORE.NormalPOS;
import massive.clearsight.DL4JDemoSentiment.CORE.NormalPOS.Results;
import massive.clearsight.linguistic_resources.ResourceDefinitions;
import massive.clearsight.utils.FileLoader;
import massive.clearsight.utils.Mathemat;
import massive.clearsight.utils.ReadWriteFile;
import massive.clearsight.utils.RegExp;
import massive.clearsight.utils.StringUtils;
import massive.clearsight.utils.TimeUtility;

/**
  * @author paolo
  * 
  * 
  * 1. The more important index is num Ones / all data (review) values, that is the density of the review info, secondly the distribution:
  * an uniform distr is better than concentrated . And the first one has entropy lower than the second, that is more predictable. In general, all zeroes and 
  * all ones has low approx entropy because the diff of the sub vectors are 0 , and this issue is learned as predictability (little noise) with respect to a
  * random distribution in which the diff increases.
  * 2. Try to use tfIDF at the place of 1 for the useful info, and zero for the unuseful (but the prob of approx entropy is that vect1_zeroes - vect2_zeroes = 0 
  * not because the diff is zero, but because each one are zero).
 *
 */
public class Reviews  extends ArrayList <Review> implements Serializable,  Cloneable {

	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	public static Set<String> stopList = null; 
	public static Set<String> externalNercWords = null; 
	public static Set<String> NERCSpecialized = null; 
	public static Set<String> positiveWords = null; 
	public static Set<String> negativeWords = null; 
	public static String language = ResourceDefinitions.LANGUAGE_EN;
	
	private List<String> usefulW;
	private List<String> unusefulW;
	private List<String> allW;
//	private static HashMap<Integer, Integer> maxHelpVoteForProd = new HashMap<Integer, Integer>(); // <prodID, maxHelpVote for prodID>
	static HashMap<String, Integer> numReviewsForProd = new HashMap<String, Integer>();  			// <prodID, #reviews>
	static HashMap<String, Integer> numHVotesForProd = new HashMap<String, Integer>();  				// <prodID, # Helpful votes>
	static HashMap<String, Integer> numTVotesForProd = new HashMap<String, Integer>();  				// <prodID, # Total votes>
	static HashMap<String, Integer> numReviewsForReviewer = new HashMap<String, Integer>();  			// <reviewerID, #reviews>
	static HashMap<String, Integer> numVotesForReviewer = new HashMap<String, Integer>();  				// <reviewerD, # H votes>
	static HashMap<String, FirsLast>  firstLastDateForProd = new HashMap<String, FirsLast>();  			// <prodID, <first date, last date>>  LifeTime prod
	static HashMap<String, Double>    numPercVotesForReviewer = new HashMap<String, Double>();  		// <reviewerD, perc votes+>
	static HashMap<String, Integer>  numAverageVotesForReviewer = new HashMap<String, Integer>();  		// <reviewerD, average # votes+ x review>
	public static VocabularyFields vocab = new VocabularyFields();
	private static int nDocs ;				// num docs Mobiles dataset (3.0 value of fist line of model w2v)
	private static String posDensity = "NN,NNS,NNP,NNPS,JJ,JJR,JJS,VB,VBP,VBD,VBG,VBN,VBZ";
	
	public static orderType typeOrder;
	public static boolean inv = false;
	public static enum orderType {Rating, TextLenghtNormal, Date, HelpfulVote, PercentHelpfulVote, TotalVote, PercentInfoUseful, SampEnt, ApproxEnt, Dens, NumReviewsForProd, ProdDateChain};
	public static weightType typeWeight = weightType.tfIDF;	// default
	public static enum weightType {one, tf, logTf, tfIDF, probtfIDF	};
	public static listType typeList;
	public static enum listType {by_lengh, by_votesUp, by_prodID, by_NumRev_prod};
	public static distType typeDist;
	public static enum distType {ChebyshevDistance, EuclidDistance};
	

	private static int dimEntropy = 2;
	private static int minReviewsProd = 50;						// threshold min num reviews x product to stabilize the values

	//private static double r;	// CALCULATE THIS VALUE!
	


	public Reviews() { 
	}
	
	/**
	 * All reviews, without condition
	 * 
	 * @param reviews
	 * @param USEFULWORDS_PATH
	 * @param vocab
	 * @param stopList
	 * @param externalNercWords
	 * @throws Exception
	 */
	public Reviews(List<String> reviews, String USEFULWORDS_PATH, VocabularyFields vocab, Set<String> stopList, 
			Set<String> externalNercWords, int typeParser, int minTextLen, int nDocs, Set<String> NERCSpecialized ) throws Exception { 
		
		Reviews.nDocs =  nDocs;
		Reviews.NERCSpecialized = NERCSpecialized;
		Reviews.negativeWords = negativeWords;
		Reviews.positiveWords = positiveWords;
		loadInfos ( USEFULWORDS_PATH,  vocab );
		NormalPOS normalizer = new NormalPOS(language.toLowerCase(), true, false, 50);	// Normalize for sentiment . Sentiment = true, Triple = false
		MessageDigest digest = ConfigurationSetting.digestInit("MD5");		// digest chosen

		String url = "^(https?|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]";	// Maintains unaltered email and url 
		String email = "^[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,6}$";

		int reviewID = 0;
		TimeUtility time = new TimeUtility();
		long diff;
		Map<Long, Integer> diffs = new TreeMap<Long, Integer>(Collections.reverseOrder());
		for (String review: reviews) { 
			Review r = null;
			time.start();

			switch (typeParser) { 
				case 0:	// 0 = traditional parser format DB Massive
					r = splitFieldsMassive(review , stopList, typeDist, normalizer, url, email, digest, "MD5",  minTextLen );
				break;				
				case 1:	// 1 = parser format data http://jmcauley.ucsd.edu/data/amazon/
					// reviewID is assigned automatically
					r = splitFieldsjmcauley(reviewID++, review , stopList, typeDist, normalizer, url, email, digest, "MD5",  minTextLen );
				break;
			}
			if (r != null)
				this.add(r);

			time.stop();
			diff = time.println();
			diffs.put(diff, r.getReviewID());

		}

		int numReviews = 0; 
		for (Review r: this){ 			

			if (numReviewsForProd.get(r.getProdID()) == null)
				numReviews = 0;
			else
				numReviews = numReviewsForProd.get(r.getProdID()) +1;
			numReviewsForProd.put(r.getProdID(), numReviews);
		}
		
	}
	
	/**
	 * Select review by condition listType typeList
	 * 
	 * @param reviews
	 * @param USEFULWORDS_PATH
	 * @param vocab
	 * @param stopList
	 * @param externalNercWords
	 * @param typeList
	 * @param value
	 * @throws Exception
	 */
	public Reviews(List<String> reviews, String USEFULWORDS_PATH, VocabularyFields vocab, Set<String> stopList, Set<String> externalNercWords, 
			listType typeList, distType typeDist, String value, int typeParser, int minTextLen, int nDocs, orderType typeOrderArg,  String outPathText
			, Set<String> NERCSpecialized, Set<String>  negativeWords, Set<String>  positiveWords ) throws Exception { 
		
		Reviews.NERCSpecialized = NERCSpecialized;
		Reviews.negativeWords = negativeWords;
		Reviews.positiveWords = positiveWords;
		loadInfos ( USEFULWORDS_PATH,  vocab );		
		String url = "^(https?|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]";	// Maintains unaltered email and url 
		String email = "^[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,6}$";
		NormalPOS normalizer = new NormalPOS(language.toLowerCase(), true, false, 30);	// Normalize for sentiment . Sentiment = true, Triple = false
		MessageDigest digest = ConfigurationSetting.digestInit("MD5");		// digest chosen



//		NormalPOS normalizer = new NormalPOS(language.toLowerCase());		
//		List<String> reviews = ConfigurationSetting.processNormalized (reviewsIN, language, false, "MD5", stopList, externalNercWords,  true, null, url, email) ;

		List<Review> list = new ArrayList<Review>();
		int numReviewsPerreviewer = 0; 
		int numReviews = 0;
		double numHelpVotesPerCent = 0; 
		int numHelpVotes = 0;
		int numHelpVotesProd = 0;
		int reviewID = 0;
		int numTotVotesProd;
		List<String> originalText = new ArrayList<String>(); 


		
		System.out.println("\nDim reviews = "+reviews.size()+"\n");
//		TimeUtility time = new TimeUtility();
//		long diff = 0L;
//		Map<Long, Integer> diffs = new TreeMap<Long, Integer>(Collections.reverseOrder());
		for (String review: reviews) { 
			
			Review r = null;
//			time.start();
			switch (typeParser) { 
				case 0:	// 0 = traditional parser format DB Massive
					r = splitFieldsMassive(review, stopList, typeDist, normalizer, url, email, digest, "MD5",  minTextLen );
				break;
				
				case 1:	// 1 = parser format data http://jmcauley.ucsd.edu/data/amazon/
					r = splitFieldsjmcauley(reviewID++, review , stopList, typeDist, normalizer, url, email, digest, "MD5",  minTextLen );
				break;
			}
//			time.stop();
//			diff = time.println();
//			if (diff != 0L && r!= null)
//				diffs.put(diff, r.getReviewID());
			
			
			if (r != null) 			
				switch (typeList) {  // listType {by_lengh, by_prodID, by_NumRev_prod};		Integer			
					 case by_lengh:  if (r.getTextLength() > Integer.parseInt(value)) list.add(r);
					 break;
					 case by_votesUp:  if (r.getHelpfulVote() > Integer.parseInt(value)) list.add(r);
					 break;
					 case by_prodID:  if (r.getProdID().equals(value)) list.add(r);  // String because in some case we use asim code amazon
					 break;
					 case by_NumRev_prod: list.add(r); // The condition  numReviewsForProd.get(r.getProdID()) > value) is set in the next code, because it need process all products 
				}
					
			if (list.size() %100 == 0)
				System.out.print("+");

			if (list.size() %1000 == 0)
				System.out.println("Num reviews processed "+list.size());

		}
		
		System.out.println("\nDim list = "+list.size()+"\n");
		System.out.println("\n****************************** ORDER TIME PROCESSING NLP SECTION ***************************************\n");
		int numItem = 0;
//		for (long timer: diffs.keySet()){
//			System.out.println("ID review: "+diffs.get(timer)+" time: "+timer);
//			numItem++;
//			if (numItem > 30)
//				break;
//		}
		System.out.println("\n****************************** END ORDER TIME PROCESSING NLP SECTION ***************************************\n");
		
		for (Review r: list){ 			
			if (numReviewsForProd.get(r.getProdID()) == null)
				numReviews = 1;
			else
				numReviews = numReviewsForProd.get(r.getProdID()) +1;
			numReviewsForProd.put(r.getProdID(), numReviews);
			
			if (listType.by_NumRev_prod.name().equals(typeList.toString()) && numReviewsForProd.get(r.getProdID()) <=  Integer.parseInt(value)) // it's complementary of numReviewsForProd.get(r.getProdID()) > value
				continue;		
			
			String reviewerID = r.getReviewerID();
			if (numVotesForReviewer.get(reviewerID) == null)
				numHelpVotes = r.getHelpfulVote();
			else
				numHelpVotes = numVotesForReviewer.get(reviewerID) +r.getHelpfulVote();
			numVotesForReviewer.put(reviewerID, numHelpVotes);
			
			if (numHVotesForProd.get(r.getProdID()) == null)
				numHelpVotesProd = r.getHelpfulVote();
			else
				numHelpVotesProd = numHVotesForProd.get(r.getProdID()) +r.getHelpfulVote();
			numHVotesForProd.put(r.getProdID(), numHelpVotesProd);
			
			if (numTVotesForProd.get(r.getProdID()) == null)
				numTotVotesProd = r.getTotalVote();
			else
				numTotVotesProd = numTVotesForProd.get(r.getProdID()) +r.getTotalVote();
			numTVotesForProd.put(r.getProdID(), numTotVotesProd);
			
			if (numPercVotesForReviewer.get(r.getReviewerID()) == null)
				numHelpVotesPerCent = r.getHelpfulVotePerCent();
			else
				numHelpVotesPerCent = numPercVotesForReviewer.get(r.getReviewerID()) +r.getHelpfulVotePerCent();
			numPercVotesForReviewer.put(r.getReviewerID(), numHelpVotesPerCent);
			
			if (numReviewsForReviewer.get(r.getReviewerID()) == null)
				numReviewsPerreviewer = 1;
			else
				numReviewsPerreviewer = numReviewsForReviewer.get(r.getReviewerID()) +1;
			numReviewsForReviewer.put(r.getReviewerID(), numReviewsPerreviewer);	
			
			originalText.add(r.getText());

			
			this.add(r);
		}
		
		ReadWriteFile.saveLines (originalText, outPathText);		
		// HERE IT'S POSSIBLE TO SAVING THE TEXT

		list = new ArrayList<Review>();  // clear the arrayList 
		originalText = new ArrayList<String>();  // clear the arrayList 
		
		System.out.println("\nFinal dim of results = "+this.size()+"\n");
		
		typeOrder = typeOrderArg;		// order by prod and the by Date ascending
		inv = false;
		Collections.sort(this);

		Date oldDate = null;
		String oldProd = null;
		FirsLast f;
		
		String  startProd = this.get(0).getProdID();
		f = new FirsLast(this.get(0).getData(), null);
		firstLastDateForProd.put(startProd, f);					// open start prod

		
		for (Review r: this){ 		
			
			if (oldDate != null && oldProd != null)
				if (!r.getProdID().equals(oldProd))	{ 			
					f = new FirsLast(r.getData(), null);
					firstLastDateForProd.put(r.getProdID(), f);					// open new prod
					firstLastDateForProd.get(oldProd).lastRev = oldDate;		// close old prod
				}
			
			double votePerCent = numPercVotesForReviewer.get(r.getReviewerID());		//  %vote+ per review of the getReviewerID()
			int numRev = numReviewsForReviewer.get(r.getReviewerID());
			numPercVotesForReviewer.put(r.getReviewerID(), votePerCent/numRev);
			
			numHelpVotes = numVotesForReviewer.get(r.getReviewerID());
			numAverageVotesForReviewer.put(r.getReviewerID(), (int)((double)numHelpVotes/numRev));
			
			oldDate = r.getData();
			oldProd = r.getProdID();
		}

		firstLastDateForProd.get(oldProd).lastRev = oldDate;
		
		
//		for (Review r: list){ 			
//			
//			switch (typeList) {  // listType {by_lengh, by_prodID, by_NumRev_prod};		Integer			
//				 case by_lengh:  if (r.getTextLength() > Integer.parseInt(value)) this.add(r);
//				 break;
//				 case by_votesUp:  if (r.getHelpfulVote() > Integer.parseInt(value)) this.add(r);
//				 break;
//				 case by_prodID:  if (r.getProdID().equals(value)) this.add(r);  // String because in some case we use asim code amazon
//				 break;
//				 case by_NumRev_prod:  if (numReviewsForProd.get(r.getProdID()) >  Integer.parseInt(value)) this.add(r);	// Integer		
//				 break;
//			}
//			
//		}

		
	}
	
 
	

	private void loadInfos (String USEFULWORDS_PATH, VocabularyFields vocab ) throws Exception { 
	
		usefulW =  FileLoader.loadSortedLines(USEFULWORDS_PATH);
	    System.out.println("dim words useful "+usefulW.size());    
	    unusefulW = new ArrayList<String>();
	    allW = new ArrayList<String>();
	    for (VocabWord vWord: vocab.getWords()){
	    	allW.add(vWord.getWord());
	    	if (!usefulW.contains(vWord.getWord()))
	    		unusefulW.add(vWord.getWord());
	    }
	    System.out.println("dim words unuUseful "+unusefulW.size()+" all words "+allW.size());
	}
	
	/**
	 * calculate several weight, and for each useful word in the text set a weight (null if the word is unuseful) 
	 * 
	 * @param text
	 * @return
	 * @throws Exception
	 */
	private ArrayList<WordWeight> calculateUsefulInfos (String text) throws Exception { 		
		
		//ArrayList<Integer> data = new ArrayList<Integer>();
		ArrayList<WordWeight> dataWeight = new ArrayList<WordWeight>();
//		List<String> listWordsOne = new ArrayList<String>();
		
		HashMap<String, Integer> freq = new HashMap<String, Integer> ();
		for (String word: text.split(" ")) {
			if (usefulW.contains(word) ) {
				if (freq.get(word) == null)
					freq.put(word, 1);
				else 
					freq.put(word, freq.get(word)+1);
			} else if (usefulW.contains("#".concat(word)) ) {	// supervised tag of most useful words (it's not useful)
				if (freq.get(word) == null)
					freq.put(word, 10);
				else 
					freq.put(word, freq.get(word)+5);
			} else 
				freq.put(word, 0);
		}
			
		
		for (String word: text.split(" ")) {
			
			if (freq.get(word) > 0) {
				try {
					
					int tf = freq.get(word);
					double logTf = Math.log(1 + (double)tf);
					double IDF = Math.log((double)nDocs/(double)vocab.getWordInDoc(word));
					double probIDF = Math.log( ((double)nDocs - vocab.getWordInDoc(word))/vocab.getWordInDoc(word));
					double tfIDF = (double)tf/IDF;	
					double probtfIDF = (double)tf/probIDF;	
					dataWeight.add(new WordWeight(word, tf, logTf, tfIDF, probtfIDF) );

					
				} catch (NullPointerException e) {		// sometimes word is not present in the vocabulary because external dataset
					dataWeight.add(null);
				}
				
			} else {
				dataWeight.add(null);
			}
		
		}
		
		
		return dataWeight;

	}

	
	/**
	 * print all Data
	 * @param outPath
	 * @throws Exception
	 */
	public void printData (String outPath ) throws Exception { 
		
		StringBuffer dataLine = new StringBuffer();
		StringBuffer headerLine = new StringBuffer();
		List<String> dataLines = new ArrayList<String>(); 
		
		typeOrder = orderType.NumReviewsForProd;
		Collections.sort(this);

		
		headerLine.append("REVIEWID").append(";").append("RATING").append(";").append("NUM ONES (USEFUL WORDS)")
		.append(";").append("SAMPLE ENT").append(";").append("APPROX ENT").append(";").append("RATIO USEFUL / ALL").append(";")
		.append("% HELPFUL VOTES").append(";")
		.append("HELPFUL VOTES").append(";").append("TOTAL VOTES").append(";").append("# REVIEWS x PROD").append(";")
		.append("PROD ID").append(";").append("LENGTH REVIEW").append(";").append("CLEANED TEXT").append("\n\n");
		
//		REVIEWID	RATING	NUM ONES (USEFUL WORDS)	
//		SAMPLE ENT	APPROX ENT	RATIO USEFUL / ALL	
//		DENSITY USEFUL / ALL	% HELPFUL VOTES	% HELPFUL VOTES 2	
//		HELPFUL VOTES	TOTAL VOTES	# REVIEWS x PROD	# HELPFUL VOTE x PROD	
//		PROD ID	CLEANED TEXT


	
		for (Review r: this){ 	
			dataLine = new StringBuffer();
			dataLine.append(r.getReviewID()).append(";").append(r.getRating()).append(";").append(r.getNumInfoUseful())
			.append(";").append(String.format("%,5.4f", r.getSampEnt())).append(";").append(String.format("%,5.4f", r.getApproxEnt())).append(";").append(String.format("%,4.3f", r.getRatioUsefullAll()))
			.append(";").append(String.format("%,4.3f", r.getHelpfulVotePerCent()))
			.append(";").append(r.getHelpfulVote()).append(";").append(r.getTotalVote()).append(";").append(numReviewsForProd.get(r.getProdID()))
			.append(";").append(r.getProdID()).append(";").append(r.getTextUltimate().split(" ").length).append(";").append(r.getTextUltimate()).append("\n");
			dataLines.add(dataLine.toString());
		}

		ReadWriteFile.saveStatisticalFunctionList(outPath, headerLine, dataLines);
		
	}

	/**
	 * calculate vote reviews within a context ordered by date (vote / mean context, vote / sd context)
	 * ONLY FOR HelpfulVote()
	 * 
	 * @param windows
	 * @param i
	 * @param sd
	 * @param mean
	 * @param aroundWind	true -> around, false -> only before (left)
	 * @return
	 * @throws Exception
	 */
	private Couple getVoteContext (int windows, int i, StandardDeviation sd , Mean mean, boolean aroundWind, orderType typeOrderContext  ) throws Exception { 
		
		double centerVote = 0;
		Review r = this.get(i);
		
		
		
		 switch (typeOrderContext) {
		 
			 case PercentHelpfulVote:  
				 centerVote  = r.getHelpfulVotePerCent();
				 break;
				 
			 case HelpfulVote: 
				 centerVote = r.getHelpfulVote();
				 break;
				 
//			 case TotalVote:  
//				 vote = r.getTotalVote();

//				 break;

		 }

		
		String prodID = r.getProdID();
		int range = windows/2;				// use window even
		double meanContext = 0.0;
		double sdContext = 0.0;
		double coeffVariation = 0.0;
		
		Vector<Double> tmp = new Vector<Double>();

		if (aroundWind)	{ 		// true -> around
			
			for (int j = i - range;  j <= i + range; j++)  
				if ((i - range) > 0 && (i + range) < this.size() && j != i ){ 
					String comProdID = this.get(j).getProdID();
					if (prodID.equals(comProdID)) 							
							
						 switch (typeOrderContext) {
						 
						 case PercentHelpfulVote:  
							 tmp.add( (double)this.get(j).getHelpfulVotePerCent());
							 break;
							 
						 case HelpfulVote: 
							 tmp.add( (double)this.get(j).getHelpfulVote());
							 break;
								 
						 }
				}
			
		} else {				// false -> only before (left)
			
			for (int j = i ;  j <= i + range; j++)  
				if (i  > 0 && (i + range) < this.size() && j != i ){ 
					String comProdID = this.get(j).getProdID();
					if (prodID.equals(comProdID)) 							
						
						 switch (typeOrderContext) {
						 
						 case PercentHelpfulVote:  
							 tmp.add( (double)this.get(j).getHelpfulVotePerCent());
							 break;
							 
						 case HelpfulVote: 
							 tmp.add( (double)this.get(j).getHelpfulVote());
							 break;
							 
					 }

				}
			
		}
		
		
		Double[] doubles = tmp.toArray(new Double[tmp.size()]);
		double[] group = ArrayUtils.toPrimitive(doubles);
		meanContext = mean.evaluate(group);
		sdContext = sd.evaluate(group);			
		coeffVariation = sdContext/meanContext;
		
		Couple cou = null;

		if (meanContext > 0) 
			cou = new Couple(Math.abs((double)centerVote - meanContext), meanContext, sdContext, coeffVariation);
		else 
			cou = new Couple(0, 0, 0);

		return cou;
		
	}
	
	/**
	 * return the num of days between current review day and first review of current prod
	 * 
	 * @param i
	 * @return
	 * @throws Exception
	 */
	private int timeFromFirstreviews (int i) throws Exception { 
		
		
		Review r = this.get(i);
		Date currentRev = r.getData();
		Date firstRev = null;
		String prodID = r.getProdID();
		
//		for (Review ri: this) { 
//			if (prodID.equals(ri.getProdID())) { 
//				firstRev = ri.getData();
//				break;
//			}
//		}
		
		FirsLast f = firstLastDateForProd.get(prodID);
		firstRev = f.firstRev;
		
		if (firstRev == null)
			return 0;
		
		int days = (new FirsLast(firstRev, currentRev)).getRangeDays();
		return days;		

	}
	
	private int timeLifeProd (int i)  throws Exception { 
		
		Review r = this.get(i);
		String prodID = r.getProdID();
		FirsLast f =  firstLastDateForProd.get(prodID);
		
		if (f == null)
			return 0;

		int days = f.getRangeDays();
		return days;		

		
	}
	
	/**
	 * 
	 * @param r
	 * @param diffDim diff dim of internal vectors iterated used to calculate the Entropy
	 * @return
	 * @throws Exception
	 */
	private Map.Entry<Double,String> calculateEntropy (Review r, int diffDim, weightType typeWeight, distType typeDist)  throws Exception { 
		//System.out.println(" text "+r.getTextUltimate()+ " data useful: "+r.getDataUseful());			// debugging
		
		double[] entropy;
		TreeMap<Double,String> entropies=new TreeMap<Double,String>();  
		
		for (double coeffThreshold = 0.25; coeffThreshold < 10; coeffThreshold *= 2) { 
	//		for (int diffDim = 1; diffDim < 3; diffDim++) { 
			entropy = entropyWeighted (r.getDataUseful(), typeWeight,  typeDist, coeffThreshold, diffDim );	
			entropies.put(entropy[0],String.valueOf(coeffThreshold)+","+String.valueOf(diffDim));
	//		}
		}
		//System.out.println("ReviewID ="+r.getReviewID()+" prodID ="+r.getProdID()+" entropies1 "+entropies1+"\n entropies2 "+entropies2);
		//Map.Entry<Double,String> minEntro = entropies.firstEntry();
		
		Map.Entry<Double,String> maxEntro =  entropies.lastEntry();		
		return maxEntro;

	}

	
	/**
	 * print Selected Data grouped by a previous criteria
	 * @param outPath
	 * @throws Exception
	 */
	public void printSelectedData (String outPath, weightType typeWeight, distType typeDist, int  window, orderType typeOrderArg) throws Exception { 
		
		StringBuffer dataLine = new StringBuffer();
		StringBuffer headerLine = new StringBuffer();
		List<String> dataLines = new ArrayList<String>(); 
		
		StandardDeviation sd = new StandardDeviation();
		Mean mean = new Mean();

		
		typeOrder = typeOrderArg;
		inv = false;
		Collections.sort(this);

			// IMPORTANT!  CHECK IF THE REVIEWS ARE ORDERED BY  orderType.ProdDateChain  (first order : Product, second date)
			// if we want the reviews by Rating an and random walk build a new Ordering (first order : Product, second rating)	// <--------- better after , dinamically, in python
		
		headerLine.append("REVIEWERID").append(";").append("REVIEWID").append(";").append("DATE").append(";").append("RATING").append(";")
		.append("SAMPLE ENT MAX1").append(";").append("RATIO USEFUL / ALL").append(";").append("1/2 RATIO USEFUL / ALL").append(";").append("% HELPFUL VOTES").append(";")
		.append("HELPFUL VOTES").append(";").append("TOTAL VOTES").append(";").append("DENS POS:NVJ").append(";").append("# REVIEWS x PROD").append(";")	
		.append("# REVIEWS x REVIEWER").append(";").append(" (MEAN %H VOTES x reviews) x REVIEWER").append(";")
		.append(" # (MEAN H VOTES x reviews) x REVIEWER").append(";")
		.append(" CURRENT LIFETIME PROD (month)").append(";").append(" FULL LIFETIME PROD  (month)").append(";").append(" # H VOTES/# T VOTES x PROD").append(";")
		.append(" MEAN H VOTES x CONTEXT (WIN = 2)").append(";").append(" COEFF VARIAT H VOTES x CONTEXT (WIN = 2)").append(";").append(" DIFF (CURRENT VOTE , MEAN VOTES) (WIN = 2)").append(";")
		.append(" MEAN H VOTES x CONTEXT (WIN = 2 LEFT)").append(";").append(" COEFF VARIAT H VOTES x CONTEXT (WIN = 2 LEFT)").append(";").append(" DIFF (CURRENT VOTE , MEAN VOTES) (WIN = 2 LEFT)").append(";")
		.append(" MEAN H VOTES x CONTEXT (WIN = 4)").append(";").append(" COEFF VARIAT H VOTES x CONTEXT (WIN = 4)").append(";").append(" DIFF (CURRENT VOTE , MEAN VOTES) (WIN = 4)").append(";")
		.append(" MEAN H VOTES x CONTEXT (WIN = 4 LEFT)").append(";").append(" COEFF VARIAT H VOTES x CONTEXT (WIN = 4 LEFT)").append(";").append(" DIFF (CURRENT VOTE , MEAN VOTES) (WIN = 4 LEFT)").append(";")
		.append("DENS. COMPOUND WORDS").append(";").append("LEN TITLE/LEN TEXT").append(";")
		.append("SENT: POSNEGSUM").append(";").append("SENT: NUMPOS/ALL").append(";").append("SENT: NUMNEG/ALL").append(";").append("SENT: NUMCONTINUOUSPOS").append(";").append("SENT: NUMCONTINUOUSNEG").append(";")
		.append("SENT 1/2: POSNEGSUM").append(";").append("SENT 1/2: NUMPOS/ALL").append(";").append("SENT 1/2: NUMNEG/ALL").append(";")
		.append("PROD ID").append(";").append("LENGTH REVIEW").append(";").append("LENGTH NORMAL REVIEW").append(";")
		.append("# POSITIVE WORDS").append(";").append("# NEGATIVE WORDS").append(";").append("ARI").append(";").append(" TEXT").append("\n");
		

		
		System.out.println("\n Calculate data for csv : "+outPath+".typeDist."+typeDist+".typeWeight."+typeWeight);
    	SimpleDateFormat  formatDate = new SimpleDateFormat("yyyy-MM-dd");  

    	for (int i = 0; i < this.size(); i++){
    		Review r = this.get(i);
			dataLine = new StringBuffer();	
			
			if (numReviewsForProd.get(r.getProdID()) < minReviewsProd)
				continue;
			
			Map.Entry<Double,String> maxEntro1 = calculateEntropy (r, 1, typeWeight, typeDist) ;
			//Map.Entry<Double,String> maxEntro2 = calculateEntropy (r, 2, typeWeight, typeDist) ;
			Couple cou1 = getVoteContext (window, i,  sd ,  mean, true,  orderType.PercentHelpfulVote) ; 		// NEW FEATUREs (4)
			Couple cou2 = getVoteContext (window, i,  sd ,  mean, false ,  orderType.PercentHelpfulVote) ; 		// NEW FEATUREs (4)
//			Couple cou3 = getVoteContext (window+1, i,  sd ,  mean, true ,  orderType.PercentHelpfulVote) ; 		// NEW FEATUREs (4)
//			Couple cou4 = getVoteContext (window+1, i,  sd ,  mean, false ,  orderType.PercentHelpfulVote) ; 		// NEW FEATUREs (4)
			Couple cou5 = getVoteContext (window+2, i,  sd ,  mean, true ,  orderType.PercentHelpfulVote) ; 		// NEW FEATUREs (4)
			Couple cou6 = getVoteContext (window+2, i,  sd ,  mean, false ,  orderType.PercentHelpfulVote) ; 		// NEW FEATUREs (4)
			
			double numPercHVotesForProd = (numTVotesForProd.get(r.getProdID()) > 0) ? (double) numHVotesForProd.get(r.getProdID())/numTVotesForProd.get(r.getProdID()) : 0;
			
			int numDays = timeFromFirstreviews (i);							// NEW FEATURE:  time distance between the current review and the first review of the same product
			int lifeTimeDays = timeLifeProd (i); 							// NEW FEATURE:  life time of Prod
			

			//System.out.println("Review "+i+"/"+this.size()+" prodID "+r.getProdID()+" date: "+formatDate.format(r.getData()));

			dataLine.append(r.getReviewerID()).append(";").append(r.getReviewID()).append(";").append(formatDate.format(r.getData())).append(";").append(r.getRating()).append(";")
			.append(String.format("%,5.4f", maxEntro1.getKey())).append(";").append(String.format("%,4.3f", r.getRatioUsefullAll())).append(";").append(String.format("%,4.3f", r.getRatioUsefullAllHalf())).append(";").append(String.format("%,4.3f", r.getHelpfulVotePerCent())).append(";")
			.append(r.getHelpfulVote()).append(";").append(r.getTotalVote()).append(";").append(String.format("%,4.3f",r.getDens())).append(";").append(numReviewsForProd.get(r.getProdID())).append(";")
			.append(numReviewsForReviewer.get(r.getReviewerID())).append(";").append(String.format("%,5.4f", numPercVotesForReviewer.get(r.getReviewerID()))).append(";")
			.append(numAverageVotesForReviewer.get(r.getReviewerID())).append(";")
			.append(String.format("%,4.3f",(double)numDays/30)).append(";").append(String.format("%,4.3f",(double)lifeTimeDays/30)).append(";").append(String.format("%,4.3f",numPercHVotesForProd)).append(";")
			.append(String.format("%,4.3f", cou1.meanContext)).append(";").append(String.format("%,4.3f", cou1.coeffVariation)).append(";").append(String.format("%,4.3f", cou1.diffMeanContext)).append(";")
			.append(String.format("%,4.3f", cou2.meanContext)).append(";").append(String.format("%,4.3f", cou2.coeffVariation)).append(";").append(String.format("%,4.3f", cou2.diffMeanContext)).append(";")
			.append(String.format("%,4.3f", cou5.meanContext)).append(";").append(String.format("%,4.3f", cou5.coeffVariation)).append(";").append(String.format("%,4.3f", cou5.diffMeanContext)).append(";")
			.append(String.format("%,4.3f", cou6.meanContext)).append(";").append(String.format("%,4.3f", cou6.coeffVariation)).append(";").append(String.format("%,4.3f", cou6.diffMeanContext)).append(";")
			.append(r.getSpecialRelative()).append(";").append(String.format("%,4.3f",r.getRatioLTitleLText())).append(";")
			.append(r.getSenti().PosNegSum).append(";").append(String.format("%,4.3f",r.getSenti().NumPositivesAll)).append(";").append(String.format("%,4.3f",r.getSenti().NumNegativesAll)).append(";").append(r.getSenti().MaxNumContiguousPositives).append(";").append(r.getSenti().MaxNumContiguousNegatives).append(";")
			.append(r.getSentiHalf().PosNegSum).append(";").append(String.format("%,4.3f",r.getSentiHalf().NumPositivesAll)).append(";").append(String.format("%,4.3f",r.getSentiHalf().NumNegativesAll)).append(";")
			.append(r.getProdID()).append(";").append(r.getTextLength()).append(";").append(r.getTextUltimate().split(" ").length).append(";")
			.append(r.getPositive()).append(";").append(r.getNegative()).append(";").append(String.format("%,4.3f",r.getAri())).append(";").append(r.getText()).append("\n");

			dataLines.add(dataLine.toString());
		}

    	ReadWriteFile.saveStatisticalFunctionList(outPath, headerLine, dataLines);		
 
		
	}
	
	
	
	private Double[]  normalize (orderType typeOrder) {
		
		Reviews.typeOrder = typeOrder;
		Collections.sort(this);
		Double[] dataArr = new Double[this.size()];
		
		for (int i = 0; i < this.size(); i++)  {
			 switch (typeOrder) {
			 
				 case PercentInfoUseful: dataArr[i] = this.get(i).getRatioUsefullAll(); 
				 break;
				 case SampEnt: dataArr[i] = this.get(i).getSampEnt(); 
				 break;
				 case ApproxEnt: dataArr[i] = this.get(i).getApproxEnt(); 
				 break;
				 default:
				 break;

			 }
		 }
		

		Double Max = dataArr[dataArr.length-1];
		//rtest = this.get(dataArr.length-1);	// approx entropy = infinite Review [ReviewID=54990,  TextLength 3, Rating=5,  Data=Thu Sep 22 00:00:00 CEST 2016, HelpfulVotePerCent=1.0, HelpfulVotePerCent2=null, HelpfulVote=1, TotalVote=1, RatioUsefullAll=1.0, DensityUsefullAll=0.0, ProdID=333, SampEnt=0.0, ApproxEnt=NaN, Title=great phone, TextUltimate= impressed problem samsungs]
		
		Double min = dataArr[0];
		for (int i = 0; i < this.size(); i++)  
			 switch (typeOrder) {
			 
				 case PercentInfoUseful: dataArr[i] = 5*(this.get(i).getRatioUsefullAll()- min) / (Max - min);
				 break;
				 case SampEnt: dataArr[i] =  5*(this.get(i).getSampEnt()- min) / (Max - min);
				 break;
				 case ApproxEnt: dataArr[i] = 5*(this.get(i).getApproxEnt()- min) / (Max - min);
				 break;
				 default:
				 break;


			 }

			
		return dataArr;
	}

	public void computeEntropy (weightType typeWeight, distType typeDist) throws Exception {
		
		double[] entropy;
		for (int i = 0; i < this.size(); i++)  {
			 entropy = entropyWeighted (this.get(i).getDataUseful(), typeWeight,  typeDist, 1, 1 );
			 this.get(i).setSampEnt(entropy[0]);
			 this.get(i).setApproxEnt(entropy[1]);
		}
	}

		
	/**
	 * X-AXIS date: Y-axis sample entropy, HelpfulVotePerCent
	 * USEFUL for understand if there is a correlation or Not
	 * 
	 */
	public void trendToHelpfulVoteOrderByDate (orderType typeOrder) {

		Reviews.typeOrder = typeOrder;
		Collections.sort(this);

		
		 switch (typeOrder) {
		 
			 case Date:
					
					System.out.println("\n HelpfulVotePerCent trend day by day;");
					for (Review r: this){
						System.out.print(String.format("%4.3f",r.getHelpfulVotePerCent())+";");
					}				
					System.out.println("\n HelpfulVote trend day by day;");
					for (Review r: this){
						System.out.print(String.format("%d",r.getHelpfulVote())+";");
					}
					System.out.println("\n sample entropy trend day by day;");
					for (Review r: this){
						System.out.print(String.format("%4.3f",r.getSampEnt())+";");
					}					
					System.out.println("\n trend day by day;");
					for (Review r: this){
						System.out.print(new SimpleDateFormat("yyyy-MM-dd").format(r.getData())+";");
					}

					System.out.println();				
					break;
					
			 case SampEnt: 
					
					System.out.println("\n HelpfulVotePerCent trend day by day;");
					for (Review r: this){
						System.out.print(String.format("%4.3f",r.getHelpfulVotePerCent())+";");
					}					
					System.out.println("\n HelpfulVote trend day by day;");
					for (Review r: this){
						System.out.print(String.format("%d",r.getHelpfulVote())+";");
					}
					System.out.println("\n sample entropy trend day by day;");
					for (Review r: this){
						System.out.print(String.format("%4.3f",r.getSampEnt())+";");
					}

					System.out.println();				
					break;
					
	

			 
		 }


		System.out.println("\n");

	}


	/**
	 * PRINT TRENDS x-axis rating step "+i+" Trend y-axis HelpfulVotePerCent 
	 */
	public void trendToHelpfulVote (orderType typeOrder, weightType typeWeight ) throws Exception {
		
		double[][] y_axis = new double[5][5];		// 5 = { [rating -> HelpfulVotePerCent][PercentInfoUseful -> HelpfulVotePerCent][...] } 5 = { segments }
		double[][] y2_axis = new double[5][5];		// 5 = { [rating -> HelpfulVote][PercentInfoUseful -> HelpfulVote][...] } 5 = { segments }
		int[][] occur_y_axis = new int[5][5];
		
		
		Reviews.typeOrder = typeOrder;
		Collections.sort(this);

		
		 switch (typeOrder) {
		 
		 case Rating:
			 

				for (Review r: this)	
					for (int i = 0; i < 5; i++) {						
						if (r.getRating() == i+1){
							y_axis[0][i] += r.getHelpfulVotePerCent();
							y2_axis[0][i] += r.getHelpfulVote();
							occur_y_axis[0][i]++;
							break;
						}
						
					}

				System.out.println("\n --------------------------- HelpfulVotePerCent and HelpfulVotes by rating   -----------------------------------------------\n");


				for (int i = 0; i < 5; i++)		{
					System.out.println(" x-axis rating info ["+i+"] y-axis volume HelpfulVotePerCent = "+String.format("%4.3f", y_axis[0][i]));
				}
				System.out.println();
				for (int i = 0; i < 5; i++)		{
					System.out.println(" x-axis rating info ["+i+"] y-axis Average HelpfulVotePerCent = "+String.format("%4.3f",(y_axis[0][i]/occur_y_axis[0][i])));
				}
				System.out.println();
				for (int i = 0; i < 5; i++)		{
					System.out.println(" x-axis rating info ["+i+"] y-axis Average HelpfulVotes = "+String.format("%4.3f", (y2_axis[0][i]/occur_y_axis[0][i])));
				}

				break;

		 
		 case PercentInfoUseful: 
			 
			    Double[] dataArrPercentInfoUsefu =  normalize (orderType.PercentInfoUseful) ;	
				System.out.println(" x-axis % useful info ->  Trend y-axis HelpfulVotePerCent ");
				for (int j = 0; j < dataArrPercentInfoUsefu.length; j++) {
					System.out.print(dataArrPercentInfoUsefu[j]+" - "+this.get(j).getHelpfulVotePerCent()+";");
					for (int i = 0; i < 5; i++) {
						if (dataArrPercentInfoUsefu[j] >= i && dataArrPercentInfoUsefu[j] < i+1){
							y_axis[1][i] += this.get(j).getHelpfulVotePerCent();
							y2_axis[1][i] += this.get(j).getHelpfulVote();
							occur_y_axis[1][i]++;
							break;
						}

					}
				}
				System.out.println("\n --------------------------- HelpfulVotePerCent and HelpfulVotes by  % useful info   -----------------------------------------------\n");

				
				for (int i = 0; i < 5; i++)		{
					System.out.println(" x-axis % useful info range ["+i+","+(i+1)+"] y-axis volume HelpfulVotePerCent = "+String.format("%4.3f", y_axis[1][i]));
				}
				System.out.println();
				for (int i = 0; i < 5; i++)		{
					System.out.println(" x-axis % useful info range ["+i+","+(i+1)+"] y-axis Average HelpfulVotePerCent = "+String.format("%4.3f", (y_axis[1][i]/occur_y_axis[1][i])));
				}
				System.out.println();
				for (int i = 0; i < 5; i++)		{
					System.out.println(" x-axis % useful info range ["+i+","+(i+1)+"] y-axis Average HelpfulVotes = "+String.format("%4.3f", (y2_axis[1][i]/occur_y_axis[1][i])));
				}

		 break;
		 
		 case SampEnt: 
			 
			 	Double[] dataArrSampEnt;
			 	dataArrSampEnt =  normalize (orderType.SampEnt) ;	
				
				System.out.println(" x-axis sample entropy ->  Trend y-axis HelpfulVotePerCent ");
				for (int j = 0; j < dataArrSampEnt.length; j++) {
					System.out.print(dataArrSampEnt[j]+" - "+this.get(j).getHelpfulVotePerCent()+";");
					for (int i = 0; i < 5; i++) {
						if (dataArrSampEnt[j] >= i && dataArrSampEnt[j] < i+1){
							y_axis[2][i] += this.get(j).getHelpfulVotePerCent();
							y2_axis[2][i] += this.get(j).getHelpfulVote();
							occur_y_axis[2][i]++;
							break;
						}

					}
				}
				System.out.println("\n --------------------------- HelpfulVotePerCent and HelpfulVotes by  sample entropy ("+typeWeight+") -----------------------------------------------\n");

				
				for (int i = 0; i < 5; i++)		{
					System.out.println(" x-axis % sample entropy range ["+i+","+(i+1)+"] y-axis volume HelpfulVotePerCent = "+String.format("%4.3f", y_axis[2][i]));
				}
				System.out.println();
				for (int i = 0; i < 5; i++)		{
					System.out.println(" x-axis % sample entropy range ["+i+","+(i+1)+"] y-axis Average HelpfulVotePerCent = "+String.format("%4.3f", (y_axis[2][i]/occur_y_axis[2][i])));
				}
				System.out.println();
				for (int i = 0; i < 5; i++)		{
					System.out.println(" x-axis % sample entropy range ["+i+","+(i+1)+"] y-axis Average HelpfulVotes = "+String.format("%4.3f", (y2_axis[2][i]/occur_y_axis[2][i])));
				}


		 break;
		 
		 case ApproxEnt: 
			 
			 	Double[] dataArrApproxEnt;
		 		dataArrApproxEnt =  normalize (orderType.ApproxEnt) ;	
				
				System.out.println(" x-axis aprox entropy ->  Trend y-axis HelpfulVotePerCent ");
				for (int j = 0; j < dataArrApproxEnt.length; j++) {
					System.out.print(dataArrApproxEnt[j]+" - "+this.get(j).getHelpfulVotePerCent()+";");
					for (int i = 0; i < 5; i++) {
						if (dataArrApproxEnt[j] >= i && dataArrApproxEnt[j] < i+1){
							y_axis[3][i] += this.get(j).getHelpfulVotePerCent();
							y2_axis[3][i] += this.get(j).getHelpfulVote();
							occur_y_axis[3][i]++;
							break;
						}

					}
				}
				System.out.println("\n --------------------------- HelpfulVotePerCent and HelpfulVotes by  aprox entropy ("+typeWeight+") -----------------------------------------------\n");

				
				for (int i = 0; i < 5; i++)		{
					System.out.println(" x-axis % aprox entropy range ["+i+","+(i+1)+"] y-axis volume HelpfulVotePerCent = "+String.format("%4.3f", y_axis[3][i]));
				}
				System.out.println();
				for (int i = 0; i < 5; i++)		{
					System.out.println(" x-axis % aprox entropy range ["+i+","+(i+1)+"] y-axis Average HelpfulVotePerCent = "+String.format("%4.3f", (y_axis[3][i]/occur_y_axis[3][i])));
				}
				System.out.println();
				for (int i = 0; i < 5; i++)		{
					System.out.println(" x-axis % aprox entropy range ["+i+","+(i+1)+"] y-axis Average HelpfulVotes = "+String.format("%4.3f", (y2_axis[3][i]/occur_y_axis[3][i])));
				}

		 break;
		 
		 
		 default:
		 break;
		 
		 }

		
		System.out.println();

	}


	public void listItemOrderedbyRating (boolean inv) {
		
		  
		  typeOrder = orderType.Rating;
		  Reviews.inv = inv;
		  Collections.sort(this);
	      for (Review r: this){
	    	  System.out.println(r);
	      }

	}
	
	
	

	public void listItemOrdered (boolean inv, orderType typeOrder) {
		
		  
		  Reviews.inv = inv;
		  Collections.sort(this);
	      for (Review r: this){
	    	  System.out.println(r);
	      }

	}
	
	
	private double[] entropyWeighted (ArrayList<WordWeight> dataUseful, weightType typeWeight, distType typeDist, double coeffThreshold, int diffDim  ) throws Exception  {
		
		Double[] dataWeighted = new Double[dataUseful.size()];
		double r = 1.0;
		
		for (int i = 0; i < dataUseful.size(); i++)
			switch (typeWeight) {  // weightType  {one, tf, logTf, tfIDF, probtfIDF	};		
			
			 case one: dataWeighted[i] = (double) ((dataUseful.get(i) != null) ? 1 : 0);
			 break;
			 case tf: dataWeighted[i] = (double) ((dataUseful.get(i) != null ) ? dataUseful.get(i).getTf() : 0); 			 
			 break;
			 case logTf: dataWeighted[i] = (double) ((dataUseful.get(i) != null) ? dataUseful.get(i).getLogTf() : 0); 
			 break;
			 case tfIDF: dataWeighted[i] = (double) ((dataUseful.get(i) != null) ? dataUseful.get(i).getTfIDF(): 0); 
			 break;
			 case probtfIDF: dataWeighted[i] = (double) ((dataUseful.get(i) != null) ? dataUseful.get(i).getProbtfIDF() : 0); 
			 break;
			 }
		
		
		
		
		if (typeWeight.equals("one"))
			r = 1;
		else {
			//System.out.println( Arrays.asList(dataWeighted));  
			r = 0.2 * Mathemat.sd(dataWeighted);
		}
		
		double[] entropy = new double[2];
		switch (typeDist) {  // distType {ChebyshevDistance, EuclidDistance};			
		 case ChebyshevDistance:  		
			 
				entropy[0] = SampEntropy.entropy(dimEntropy, diffDim, coeffThreshold*r, dataWeighted, distType.ChebyshevDistance);
				entropy[1] = ApproxEntropy.entropy(dimEntropy, coeffThreshold*(r/2), dataWeighted, distType.ChebyshevDistance);
				return entropy;

		 case EuclidDistance:  			 
				entropy[0] = SampEntropy.entropy(dimEntropy, diffDim, coeffThreshold*r, dataWeighted, distType.EuclidDistance);
				entropy[1] = ApproxEntropy.entropy(dimEntropy, coeffThreshold*(r/2), dataWeighted, distType.EuclidDistance);
				return entropy;

		 default:  
				entropy[0] = SampEntropy.entropy(dimEntropy, diffDim, coeffThreshold*r, dataWeighted, distType.EuclidDistance);
				entropy[1] = ApproxEntropy.entropy(dimEntropy, coeffThreshold*(r/2), dataWeighted, distType.EuclidDistance);
				return entropy;

		}
		
	}

	
	
	/**
	 * calculate the percentage of N,J,V with respect to the POS of the text
	 * @param tokens  :  String token = lemma + '{' + pos + '}';
	 * @param NJV  :  "NN,NNS,NNP,NNPS,JJ,JJR,JJS,VB,VBP,VBD,VBG,VBN,VBZ"
	 * @return
	 */
	private double density (List<String> tokens, String NJV)  {
		
		Set<String>enabledPos =StringUtils.stringToStringSet(NJV.toLowerCase());	
		int value = 0;
		
		for (String tokenPOS: tokens) {
			
			String pos = StringUtils.getTagFromTaggedString(tokenPOS).toLowerCase() ;
			if (enabledPos.contains(pos))
				value++;
		}

		// calculate percentage
		
		return ((double)value)/tokens.size();
		
	}
	
	
	/**
	 * 
	 * MASSIVE FORMAT
	 * 
	 * @param review
	 * @return
	 * @throws Exception
	 * 
	 * Es.: 1######samsung######samsung######5######i like samsung products bt i really like this laptop. i got it for my start up business and so far so good it has such a clear screen and i love the way u can bounce from the 
	 * keyboard to touch screen without having to push any buttons to change it i'm really excited to get to learn more on my samsung laptop!!! :) and further my career using it!
	 * ######samsung{vbn} product{nns} bt{vbp} laptop{nn} ## start_up{nerc} business{nn} so_good{nerc} 1{z} clear{jj} screen{nn} love{vbp} bounce{vb} keyboard{nn} touch_screen{nerc} without{in} push{vb} button{nns} change{vb} excite{vbn} 
	 * learn{vb} samsung{vbn} laptop{nn} !{fat} !{fat} !{fat} :{fd} ){fpt} career{nn}######2017-03-27######https://4166582898635501447.com######9######9######3######1###
	 * 
	 * 2######Fantastic all around laptop.######fantastic around laptop######5######Definitely get this version over the $200 more version with the ssd. the difference in price will get you a much bigger and better ssd 
	 * (which you absolutely NEED for any computer in todays world) and will make geek squad cheaper as well. The screen is by far one of the best displays on a laptop even at 1080p. It doesn't wash out at any angle like most 
	 * laptops and the macbooks do. The keyboard is well spaced. The hinge is solid. YOU NEED TO UPGRADE TO AN SSD though. This laptop has a m.2 sata3 port inside so you can add an ssd instead of swapping out the entire hdd. 
	 * This laptop SCREAMS with an ssd inside. With windows 10 it takes me all of 3 seconds from off to booted (vs 45 seconds to a minute 30) and another half second (vs 15-20 sec) to open the browser. The computer comes with 12gb 
	 * of ddr4 ram (upgradeable to 32gb with two 16gb sticks) which is difficult to access but is user swappable (just like the hard drive) It will play newer big name games at pretty bad framerates but doesn't do too horrible after you 
	 * overclock the graphics chip a little, however the laptop is built more for video editing than video game playing.######version{NN} 200{Z} version{NN} ssd{NN} .{Fp} difference{NN} price{NN} get_you{NERC} 1{Z} big{JJR} ssd{VBD} ({Fpa} 
	 * absolutely{RB} computer{NN} today{NNS} world{NN} ){Fpt} make{VB} geek_squad{NERC} cheap{JJR} ## screen{NN} good{JJS} display{NNS} 1{Z} laptop{NN} 1080p{Z} ## not{RB} wash{VB} angle{NN} laptop{NNS} macbooks{NNS} ## keyboard{NN} 
	 * space{VBN} ## hinge{NN} solid{JJ} ## upgrade{VB} 1{Z} ssd{NN} ## laptop{NN} 1{Z} m2{Z} sata3{Z} port{NN} add{VB} 1{Z} ssd{NN} instead_of{NERC} swap{VBG} entire{JJ} hdd{JJ} ## laptop{JJ} scream{NNS} 1{Z} ssd{JJ} ## window{NNS} 
	 * 10{Z} 3{Z} boot{VBD} ({Fpa} 45{Z} 1{Z} minute{NN} 30{Z} ){Fpt} half{NN} 2{JJ} ({Fpa} 15_20{Z} sec{NN} ){Fpt} open{VB} browser{NN} ## computer{NN} 12gb{Z} ddr4{Z} ram{NN} ({Fpa} upgradeable{JJ} 32gb{Z} 2{Z} 16gb{Z} stick{NNS} ){Fpt} 
	 * difficult{JJ} access{NN} user{NN} swappable{JJ} ({Fpa} hard_drive{NERC} ){Fpt} play{VB} big{JJ} game{NNS} pretty{RB} bad{JJ} framerates{VBZ} not{RB} horrible{JJ} overclock{NN} graphic{NNS} chip{NN} 1{Z} little{JJ} ,{Fc} 
	 * laptop{NN} build{VBN} video_edit{NERC} than{IN} video_game{NERC} play{VBG}######2017-06-07######https://1254182675966575520.com######5######5######3######2###

	   case. without normalized title	
       110960######the######1######purchased unit and after turning i t on found there was one week left on warranty l from hp.. this was an old unit. contacted party and was told to buy an aftermarket warranty and they would reimburse. no thanks, the box also showed walmart decals on the outside!######purchase{vbn} unit{nn} turn{vbg} find{vbd} week{nn} leave{vbn} warranty{nn} hpbrand{nn} .{fp} ## 1{z} unit{nn} ## contact{vbn} party{nn} buy{vb} 1{z} aftermarket{nn} warranty{nn} reimburse{vb} ## not{dt} ,{fc} box{nn} show{vbd} walmart{jj} decal{nns}######2017-09-10######https://www.amazon.com/gp/customer-reviews/r3090j5twycxk8######0######0######4773######110415

	 * Campi limitati da: ###
	 * 
	 */
     public Review splitFieldsMassive(String review, Set<String> stopList, distType typeDist,  NormalPOS normalizer, String url, String email, 
    		MessageDigest digest, String digestAlgorithm, int minTextLen ) throws Exception { 
    	
    	
    	
    	String fields[] = review.split("######");
    	//System.out.println(review);
 
    	
     	try {
    		
		    	Integer reviewID = Integer.parseInt(fields[0]);
		    	String title = fields[1];
		    	// titleNormal 2
		    	Integer rating = Integer.parseInt(fields[3]);
		    	String text = fields[4];
		    	Integer textLength = RegExp.countElements (" ", text ) + 1;		// num words
		    	Integer textLengthChar = text.length();		// num chars
		    	Integer numSentences = RegExp.countElements ("(\\.|\\?|\\!)", text );		// num sentences
		    	Double ARI = 4.71 * ((double)(textLengthChar - textLength)/(double)textLength) + 0.5 * ((double)textLength/(double)numSentences) - 21.43;
		    	if (ARI < 0 )
		    		ARI = 0.0;
		    	else if (ARI > 25)
		    		ARI = 25.0;


		    	
		    	String preNormal = ConfigurationSetting.preFiltering1 (text);
		    	//String in = preNormal.toString().trim();
		    	Results res = normalizer.normalizePOSdirectPlusSenti (preNormal);
				List<String> tokens = res.taggedLemmas;
				List<Integer>  sentiment =  res.sentenceSentiment;
				Sentiment senti = new Sentiment (sentiment);
				List<Integer>  sentimentHalf = sentiment.subList(0, (sentiment.size() >= 6) ? 6 : sentiment.size());
				//List<Integer>  sentimentHalf =  normalizer.sentiment2(preNormal.substring(0, preNormal.length()/2));
				Sentiment sentiHalf = new Sentiment (sentimentHalf);

//				List<String> tokens = normalizer.normalizePOSPlusTriplesdirect(preNormal.toString().trim());		// <-------------------try this
				Double den = density (tokens, posDensity);		// percentage of N,J,V    

				List<String> postNormalTokens = ConfigurationSetting.postFiltering (tokens, url,  email, false, null,  null );
				String linePostNerc = ConfigurationSetting.filterLine(postNormalTokens, externalNercWords, stopList, digest, digestAlgorithm, true) ;
				if (linePostNerc == null || linePostNerc.isEmpty())
					return null;
		
				// linePostNerc !!!!!!!!!! check
				String textUltimate = linePostNerc.trim();	// Not delete stop words!!!
//				for (String w: linePostNerc.split(" "))
//					if (!stopList.contains(w))
//						textUltimate = textUltimate.concat(w).concat(" ");

				Integer textLengthNormal = RegExp.countElements (" ", textUltimate ) + 1;
		    	if (textLengthNormal < minTextLen)
		    		return null;
		    	
		    	int specialize = 0;
		    	int positive = 0;
		    	int negative = 0;
				for (String w: textUltimate.split(" ")) {
					w = w.replaceAll("_", " ");
					if (Reviews.NERCSpecialized.contains(w))
						specialize++;
				}
				for (String w: textUltimate.split(" ")) {
//					w = w.replaceAll("_", " ");
					if (Reviews.positiveWords.contains(w))
						positive++;
					if (Reviews.negativeWords.contains(w))
						negative++;
				}
				Double specialRelative = ((double)specialize)/textLengthNormal;
				Integer titleLength = RegExp.countElements (" ", title ) + 1;		// num words
				double ratioLTitleLText = (double)titleLength/textLength;

		    	// POS 6
		    	// NULL 7
		    	// String dtStart = "2017-06-26";  
		    	SimpleDateFormat  format = new SimpleDateFormat("yyyy-MM-dd");  
		    	Date date = format.parse(fields[6].trim());
		    	String URL = fields[7].trim();
		    	Integer helpfulVote = Integer.parseInt(fields[8]);
		    	Integer allVote = Integer.parseInt(fields[9]);
		    	Double HelpfulVotePerCent = 0.0;
		    	if (allVote > 0)
		    		HelpfulVotePerCent = ((double)helpfulVote.intValue())/allVote;
		    	String prodID = fields[10];
		    	// ReviewerID fields[11];
		    	// pattern of Useful and Unuseful INFO (calculate several weights: tfIDF, logTf, ...) 
		    	ArrayList<WordWeight> dataUseful = calculateUsefulInfos (textUltimate);		//for each useful word in the text set a weight (null if the word is unuseful) 
		     	//r = 0.2 * Mathemat.sd(dataUseful);	
				int numInfoUseful = 0;
				for (int i = 0; i < dataUseful.size(); i++)
					if (dataUseful.get(i) != null)
						numInfoUseful++;
				Double ratioUsefullAll = ((double)numInfoUseful)/dataUseful.size();			// ratio between useful and total info
				
				ArrayList<WordWeight> dataUsefulHalf = calculateUsefulInfos (textUltimate.substring(0, textUltimate.length()/2));		//for each useful word in the text set a weight (null if the word is unuseful) 
				int numInfoUsefulHalf = 0;
				for (int i = 0; i < dataUsefulHalf.size(); i++)
					if (dataUsefulHalf.get(i) != null)
						numInfoUsefulHalf++;
				Double ratioUsefullAllHalf = ((double)numInfoUsefulHalf)/dataUsefulHalf.size();			// ratio between useful and total info

				
				Double sampEnt =  0.0;
				Double approxEnt = 0.0;
		    	if (dataUseful.size() > dimEntropy*4) {  //  for little text the entropy calculus became infinite or zero
		    		
		    		// The entropy here is only a default setting, because in the next operation they will be changed the values
		     		double[] entropy =  entropyWeighted (dataUseful, weightType.tfIDF, typeDist, 1, 1 );
		    		sampEnt = entropy[0];			
		        	approxEnt = entropy[1];			
		    	}
		    	
		    	// Add double ratioLTitleLText
		    	// Add class sentiment  Sentiment senti
		    	// Double ratioUsefullAllHalf
		    	// Sentiment sentiHalf
		    	
		    	
		    	Review r = new Review("", reviewID, title, rating, text, textUltimate, textLengthNormal, textLength, date, "", HelpfulVotePerCent, helpfulVote, allVote, numInfoUseful, 
		    			ratioUsefullAll, ratioUsefullAllHalf, dataUseful, prodID, sampEnt, approxEnt, den, specialRelative, positive, negative, ratioLTitleLText, senti, sentiHalf, ARI  );

		    	
		    	return r;
    	
		} catch (NumberFormatException e) {   		
		   	System.out.println("ERROR FORMAT "+review);
		   	return null;
		} catch (NullPointerException e) {       		
		   	System.out.println("ERROR FORMAT "+review);    		
		   	return null;
		} catch (Exception e) {       		
		   	System.out.println("ERROR FORMAT "+review);
		   	return null;
		}

    	
     } 

    /**
     * 
     * jmcauley AMAZON RESEARCHER FORMAT
     * 
     * Es.:{"reviewerID": "A3F73SC1LY51OO", "asin": "B00002243X", "reviewerName": "Alan Montgomery", "helpful": [4, 4], "reviewText": 
     * "I needed a set of jumper cables for my new car and these had good reviews and were at a good price.  They have been used a few times already and 
     * do what they are supposed to - no complaints there.What I will say is that 12 feet really isn't an ideal length.  Sure, if you pull up front bumper 
     * to front bumper they are plenty long, but a lot of times you will be beside another car or can't get really close.  Because of this, 
     * I would recommend something a little longer than 12'.Great brand - get 16' version though.", "overall": 5.0, 
     * "summary": "Work Well - Should Have Bought Longer Ones", "unixReviewTime": 1313539200, "reviewTime": "08 17, 2011"}

     * @param review
     * @param stopList2
     * @param typeDist2
     * @return
     */
	private Review splitFieldsjmcauley(Integer reviewID, String review, Set<String> stopList, distType typeDist,  NormalPOS normalizer, String url, String email, 
    		MessageDigest digest, String digestAlgorithm, int minTextLen ) throws Exception { 

	   	String fields[] = review.substring(1, review.length()-1).split("(\"\\, \"|\\]\\, \"|[0-9]\\, \")");		// remove start { } and split   case1= \"\\, \"  case2= \\]\\, \"  case3= [0-9]\\, \" 
	   	// using such separator ",
    	// System.out.println(review);
    	 

	   	
	   	
     	try {
    		

			   	String reviewerID = (fields[0].split("\\:"))[1];
			   	reviewerID = reviewerID.replaceAll("\"", "").trim();
			   	String asin = (fields[1].split("\\:"))[1].replace("\"", "").trim();			// asin code -> prodID  product-identification within Amazon.com organization
		
			   	int shift = 0;
			   	if ((((fields[2]).split("\\:"))[0]).toLowerCase().contains("reviewername"))
			   		shift = 1;
			   	String title = ((fields[5+shift].split("\\:"))[1]);
			   	String tmp = ((fields[4+shift].split("\\:"))[1]).replaceAll("\\.", "").trim();
		     	int  rating = Integer.parseInt(tmp);
		    	String text = ((fields[3+shift].split("\\:"))[1]);		// Simpler parsing in this way for text
		    	text = text.substring(text.indexOf("\"")+1);
		    	String vote = (fields[2+shift].split("\\:"))[1].replaceAll("\\[\\]", "").trim();	
		    	Integer helpfulVote = Integer.parseInt(vote.split("\\,")[0].replaceAll("\\[", "").trim());
		    	Integer allVote = Integer.parseInt(vote.split("\\,")[1].trim());
		    	
		    	Integer textLength = RegExp.countElements (" ", text ) + 1;		// num words
		    	Integer textLengthChar = text.length();		// num chars
		    	Integer numSentences = RegExp.countElements ("(\\.|\\?|\\!)", text );		// num sentences
		    	Double ARI = 4.71 * ((double)(textLengthChar - textLength)/(double)textLength) + 0.5 * ((double)textLength/(double)numSentences) - 21.43;
		    	if (ARI < 0 )
		    		ARI = 0.0;
		    	else if (ARI > 25)
		    		ARI = 25.0;

		    	Double HelpfulVotePerCent = 0.0;
		    	if (allVote > 0)
		    		HelpfulVotePerCent = ((double)helpfulVote.intValue())/allVote;
		    	String preNormal = ConfigurationSetting.preFiltering1 (text);
		    	//String in = preNormal.toString().trim();
		    	Results res = normalizer.normalizePOSdirectPlusSenti (preNormal);
				List<String> tokens = res.taggedLemmas;
				List<Integer>  sentiment =  res.sentenceSentiment;
				Sentiment senti = new Sentiment (sentiment);
				List<Integer>  sentimentHalf = sentiment.subList(0, (sentiment.size() >= 6) ? 6 : sentiment.size());
				//List<Integer>  sentimentHalf =  normalizer.sentiment2(preNormal.substring(0, preNormal.length()/2));
				Sentiment sentiHalf = new Sentiment (sentimentHalf);

				Double den = density (tokens, posDensity);		// percentage of N,J,V    
				
				List<String> postNormalTokens = ConfigurationSetting.postFiltering (tokens, url,  email, false, null,  null );
				String linePostNerc = ConfigurationSetting.filterLine(postNormalTokens, externalNercWords, stopList, digest, digestAlgorithm, true) ;
				if (linePostNerc == null || linePostNerc.isEmpty())
					return null;
		
				// linePostNerc !!!!!!!!!! check
				String textUltimate = linePostNerc.trim();	// Not delete stop words!!!
//				for (String w: linePostNerc.split(" "))
//					if (!stopList.contains(w))
//						textUltimate = textUltimate.concat(w).concat(" ");

				Integer textLengthNormal = RegExp.countElements (" ", textUltimate ) + 1;		// num words
		    	if (textLengthNormal < minTextLen)
		    		return null;
		    	
		    	int specialize = 0;
		    	int positive = 0;
		    	int negative = 0;
				for (String w: textUltimate.split(" ")) {
					w = w.replaceAll("_", " ");
					if (Reviews.NERCSpecialized.contains(w))
						specialize++;
				}
				for (String w: textUltimate.split(" ")) {
//					w = w.replaceAll("_", " ");
					if (Reviews.positiveWords.contains(w))
						positive++;
					if (Reviews.negativeWords.contains(w))
						negative++;
				}

				double specialRelative = (double)specialize/textLengthNormal;
				Integer titleLength = RegExp.countElements (" ", title ) + 1;		// num words
				double ratioLTitleLText = (double)titleLength/textLength;		// ratio LEN title/LEN review
				

		
		    	SimpleDateFormat  format = new SimpleDateFormat("MM dd, yyyy");  			// "08 17, 2011"
		    	Date date = format.parse((fields[7+shift].split("\\:"))[1].replace("\"", "").trim());
		    	
		
		    	// pattern of Useful and Unuseful INFO (calculate several weights: tfIDF, logTf, ...) 
		    	ArrayList<WordWeight> dataUseful = calculateUsefulInfos (textUltimate);		//for each useful word in the text set a weight (null if the word is unuseful) 
				int numInfoUseful = 0;
				for (int i = 0; i < dataUseful.size(); i++)
					if (dataUseful.get(i) != null)
						numInfoUseful++;
				Double ratioUsefullAll = ((double)numInfoUseful)/dataUseful.size();			// ratio between useful and total info
		    	
				ArrayList<WordWeight> dataUsefulHalf = calculateUsefulInfos (textUltimate.substring(0, textUltimate.length()/2));		//for each useful word in the text set a weight (null if the word is unuseful) 
				int numInfoUsefulHalf = 0;
				for (int i = 0; i < dataUsefulHalf.size(); i++)
					if (dataUsefulHalf.get(i) != null)
						numInfoUsefulHalf++;
				Double ratioUsefullAllHalf = ((double)numInfoUsefulHalf)/dataUsefulHalf.size();			// ratio between useful and total info
				
				Double sampEnt =  0.0;
				Double approxEnt = 0.0;
		    	if (dataUseful.size() > dimEntropy*4) {  //  for little text the entropy calculus became infinite or zero
		    		
		    		// The entropy here is only a default setting, because in the next operation they will be changed the values
		     		double[] entropy =  entropyWeighted (dataUseful, weightType.tfIDF, typeDist, 1, 1 );
		    		sampEnt = entropy[0];			
		        	approxEnt = entropy[1];			
		    	}
		    	
		    	// Add double ratioLTitleLText
		    	// Add class sentiment  Sentiment senti
		    	// Double ratioUsefullAllHalf
		    	// Sentiment sentiHalf
		    	
		    	Review r = new Review(reviewerID, reviewID, title, rating, text, textUltimate, textLengthNormal, textLength, date, "", HelpfulVotePerCent, helpfulVote, allVote, numInfoUseful, 
		    			ratioUsefullAll, ratioUsefullAllHalf, dataUseful, asin, sampEnt, approxEnt, den, specialRelative, positive, negative, ratioLTitleLText, senti, sentiHalf, ARI  );
		    	
		    	return r;
    	
		} catch (NumberFormatException e) {   		
			e.printStackTrace();
		   	System.out.println("ERROR FORMAT "+review);
		   	return null;
		} catch (NullPointerException e) {    
			e.printStackTrace();
		   	System.out.println("ERROR FORMAT "+review);    		
		   	return null;
		} catch (Exception e) {       		
			e.printStackTrace();
		   	System.out.println("ERROR FORMAT "+review);
		   	return null;
		}
	
    	

	}

	public static String splitFieldsjmcauley(String review ) throws Exception { 

	   	String fields[] = review.substring(1, review.length()-1).split("(\"\\, \"|\\]\\, \"|[0-9]\\, \")");		// remove start { } and split   case1= \"\\, \"  case2= \\]\\, \"  case3= [0-9]\\, \" 
	   	// using such separator ",
    	// System.out.println(review);
    	 

	   	
	   	
     	try {
    		

			   	String reviewerID = (fields[0].split("\\:"))[1];
			   	String asin = (fields[1].split("\\:"))[1].replace("\"", "").trim();			// asin code -> prodID  product-identification within Amazon.com organization
			   	return asin;
		
//			   	int shift = 0;
//			   	if ((((fields[2]).split("\\:"))[0]).toLowerCase().contains("reviewername"))
//			   		shift = 1;
//			   	String title = ((fields[5+shift].split("\\:"))[1]);
//			   	String tmp = ((fields[4+shift].split("\\:"))[1]).replaceAll("\\.", "").trim();
//		     	int  rating = Integer.parseInt(tmp);
//		    	String text = ((fields[3+shift].split("\\:"))[1]);		// Simpler parsing in this way for text
//		    	String vote = (fields[2+shift].split("\\:"))[1].replaceAll("\\[\\]", "").trim();	
//		    	Integer helpfulVote = Integer.parseInt(vote.split("\\,")[0].replaceAll("\\[", "").trim());
//		    	Integer allVote = Integer.parseInt(vote.split("\\,")[1].trim());
//		    	
//		
//		    	SimpleDateFormat  format = new SimpleDateFormat("MM dd, yyyy");  			// "08 17, 2011"
//		    	Date date = format.parse((fields[7+shift].split("\\:"))[1].replace("\"", "").trim());
		    	
    	
		} catch (NumberFormatException e) {   		
			e.printStackTrace();
		   	System.out.println("ERROR FORMAT "+review);
		   	return null;
		} catch (NullPointerException e) {    
			e.printStackTrace();
		   	System.out.println("ERROR FORMAT "+review);    		
		   	return null;
		} catch (Exception e) {       		
			e.printStackTrace();
		   	System.out.println("ERROR FORMAT "+review);
		   	return null;
		}
	
    	

	}
	
	
	private class Couple implements Serializable,  Cloneable {
		
		double  meanContext = 0.0;
		double  sdContext  = 0.0;
		double 	diffMeanContext  = 0.0;	
		double 	coeffVariation  = 0.0;	
		
		
		public Couple(double diffMeanContext, double meanContext, double sdContext) {
			super();
			this.meanContext = meanContext;
			this.sdContext = sdContext;
			this.diffMeanContext = diffMeanContext;
			
		}
		public Couple(double diffMeanContext, double meanContext, double sdContext, double coeffVariation) {
			super();
			this.meanContext = meanContext;
			this.sdContext = sdContext;
			this.diffMeanContext = diffMeanContext;
			this.coeffVariation = coeffVariation;
			
		}
		
		public Couple(double meanContext, double sdContext) {
			super();
			this.meanContext = meanContext;
			this.sdContext = sdContext;
			
		}
		@Override
		public String toString() {
			return "Couple [meanContext=" + meanContext + ", sdContext=" + sdContext + ", diffMeanContext="
					+ diffMeanContext + ", coeffVariation=" + coeffVariation + "]";
		}
		

	}

		private class FirsLast implements Serializable,  Cloneable {
			
			/**
			 * 
			 */
			private static final long serialVersionUID = 4429608129654265475L;
			Date firstRev = null;
			Date lastRev = null;
			
			public FirsLast(Date firstRev, Date lastRev) {
				super();
				this.firstRev = firstRev;
				this.lastRev = lastRev;
			}
			
			public int getRangeDays () {
				long range = lastRev.getTime() - firstRev.getTime();
				int days = (int)(range / (1000*60*60*24));
				return days;
			}

			public int getRangeMonths () {
				long range = lastRev.getTime() - firstRev.getTime();
				int months = (int)(range / (1000*60*60*24*30));
				return months;
			}

			@Override
			public String toString() {
				return "FirsLast [firstRev=" + firstRev + ", lastRev=" + lastRev + "]";
			}

			
			
		}
		

		public class Sentiment implements Serializable,  Cloneable {
			
			private static final long serialVersionUID = 442960816541265475L;
			
//			STANFORD CORE NLP : 
//				algebraic sum of positives and negatives
//				num of positive, 
//				num neutral, 
//				num negative

			Integer PosNegSum = 0;			// algebraic Sum
			Double NumPositivesAll = 0.0;	// Num Positives sentences / All sentences
			Double NumNegativesAll = 0.0;	// Num Negatives sentences / All sentences
			Integer MaxNumContiguousPositives = 0;
			Integer MaxNumContiguousNegatives = 0;
			
			Integer previousSent = 0;
			Integer MaxPlus=0, MaxMinus=0;
			
			public Sentiment (List<Integer>  sentiment)  {
				
				for (Integer sentisentence: sentiment) {
					
					PosNegSum += sentisentence;
					if (sentisentence == 1) {
						NumPositivesAll++;
						
						if (previousSent != sentisentence) {
							MaxNumContiguousPositives = 1;
						} else
							MaxNumContiguousPositives++;
						
						if (MaxNumContiguousPositives > MaxPlus)
							MaxPlus = MaxNumContiguousPositives;

					} else if (sentisentence == -1) {
						NumNegativesAll++;

						if (previousSent != sentisentence) {
							MaxNumContiguousNegatives = 1;
						} else
							MaxNumContiguousNegatives++;
						
						if (MaxNumContiguousNegatives > MaxMinus)
							MaxMinus = MaxNumContiguousNegatives;

					} else {
						MaxNumContiguousPositives = 0;
						MaxNumContiguousNegatives = 0;
					}
					
					previousSent = sentisentence;

				}
				
				NumPositivesAll = NumPositivesAll/sentiment.size();
				NumNegativesAll = NumNegativesAll/sentiment.size();
				
				MaxNumContiguousPositives = MaxPlus;
				MaxNumContiguousNegatives = MaxMinus;
				
			}

			@Override
			public String toString() {
				return "Sentiment [PosNegSum=" + PosNegSum + ", NumPositivesAll=" + NumPositivesAll
						+ ", NumNegativesAll=" + NumNegativesAll + ", MaxNumContiguousPositives="
						+ MaxNumContiguousPositives + ", MaxNumContiguousNegatives=" + MaxNumContiguousNegatives
						+ ", previousSent=" + previousSent + ", MaxPlus=" + MaxPlus + ", MaxMinus=" + MaxMinus + "]";
			}
			
			
			
		}


}
