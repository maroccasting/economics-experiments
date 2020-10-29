package massive.clearsight.Vote_Rate_Entropy;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.TreeMap;
import java.util.Vector;


import checkData.CheckData;
import massive.clearsight.Vote_Rate_Entropy.Review;
import massive.clearsight.utils.FileLoader;
import massive.clearsight.utils.Mathemat;
import massive.clearsight.utils.ReadWriteFile;

public class LoadResultAndCalculate {

	private static List<Review> patterns;
	private static TreeMap<Float, Vector<Review>> ReviewPerYear = new TreeMap<Float, Vector<Review>>() ;	
	public static enum listType {by_Prod,by_Prod2, all};
	public static enum freqType {year, month};
	public static listType typeList;
	public static listType typeFreq;
	private final static int startYear = 2012;
	private final static int endYear = 2017;
	private final static int MinNumYears = 4;
	private final static int minNumbervaluesbyMonth = 30;
	

	public static void main(String[] args) {
		
			try {
				
				String DATA_PATH = args[0];			
		
				File myFile = new File(DATA_PATH);
				String outDir = myFile.getParent();
				String name = myFile.getName();
				
				
		
				patterns =  new ArrayList<Review> ();
				List<String> data =  FileLoader.loadSortedLines(DATA_PATH); 
				int i = 0;
				for (String line: data) { 
					
					if (i > 0) {
						Review r = splitFieldsReviews(line);
						if (r != null)
							patterns.add(r);
					}
					i++;		
				}
				PrintReviewsPerYear (outDir+File.separatorChar+"ResultMean"+name+"."+args[1], listType.valueOf(args[1]), freqType.valueOf(args[2]));
				
				

			} catch (Exception e) {
				e.printStackTrace();
			}


		
	}
	
	   /**
	    * typeFreq is used only in the case by_Prod2: 
	    * 
	    * @param outPath
	    * @param typeList
	    * @param typeFreq
	    * @throws Exception
	    */
	   private static void PrintReviewsPerYear (String outPath, listType typeList, freqType typeFreq) throws Exception { 
		   
		  
		    switch (typeFreq) { 
			   	case year:   loadRevPerYear () ;
			   	break;
				   
			   	case month:  loadRevPerYear () ;
			   				 loadRevPerMonth () ;
			   	break;
		    }
		    
			List<String> dataLines = new ArrayList<String>(); 
			StringBuffer headerLine = null; 
			


		   HashSet<String> listProdID = new HashSet<String>();
		   switch (typeList) { 	
		   
			 case by_Prod:  // Each line = one product and one year
				 	headerLine = buildHeader(typeList, typeFreq, "");
					for (Review r: patterns) { 
						 listProdID.add(r.getProdID());
					}
					for (String prodID: listProdID) {
						calculateMeans(dataLines, prodID);
					}
					ReadWriteFile.saveStatisticalFunctionList(outPath, headerLine, dataLines);		
			 break;
			 case all:  
				 	headerLine = buildHeader(typeList, typeFreq, "");
					calculateMeans(dataLines, "");
					ReadWriteFile.saveStatisticalFunctionList(outPath, headerLine, dataLines);		
			 break;
			 case by_Prod2:  // Each line = one product each column one year
				 
				    @SuppressWarnings("unchecked")
				    HashMap<String, TreeMap<Float, Float>>[] SuperMaps = new HashMap[3];
				    HashMap<String, TreeMap<Float, Float>> MapPositVotes= new HashMap<String, TreeMap<Float, Float>>();		// <prodID, TreeMap<Year, PositVotesPerYear>>
				    HashMap<String, TreeMap<Float, Float>> MapTotalVotes= new HashMap<String, TreeMap<Float, Float>>();		// <prodID, TreeMap<Year, TotalVotesPerYear>>
				    HashMap<String, TreeMap<Float, Float>> MapNumReviews= new HashMap<String, TreeMap<Float, Float>>();		// <prodID, TreeMap<Year, NumReviewsPerYear>>


					for (Review r: patterns) { 
						 listProdID.add(r.getProdID());
					}
					for (String prodID: listProdID) {
						TreeMap<Float, Float>[] Maps = calculateMeans(dataLines, prodID);
						MapPositVotes.put(prodID, Maps[0]);
						MapTotalVotes.put(prodID, Maps[1]);
						MapNumReviews.put(prodID, Maps[2]);
					}
				    SuperMaps[0] = MapPositVotes;
				    SuperMaps[1] = MapTotalVotes;
				    SuperMaps[2] = MapNumReviews;
				    StringBuffer Superlines = builAllLines (SuperMaps, typeFreq);
				    ReadWriteFile.saveStatisticalFunction(outPath, Superlines);

			}


	   }
	   
	
	   private static StringBuffer builAllLines (HashMap<String, TreeMap<Float, Float>>[] SuperMaps, freqType typeFreq) throws Exception { 
		   
		    switch (typeFreq) { 
		    
		   	case year:  
				   StringBuffer Superlines = builGroupLinesByYear (0,  SuperMaps, "POSITIVE VOTES TREND", typeFreq);
				   Superlines.append(builGroupLinesByYear (1,  SuperMaps, "TOTAL VOTES TREND", typeFreq));
				   Superlines.append(builGroupLinesByYear (2,  SuperMaps, "NUM REVIEWS TREND", typeFreq));
				   return Superlines;

		   	case month:  
				   Superlines = builGroupLinesByMonth (0,  SuperMaps, "POSITIVE VOTES TREND", typeFreq);
				   Superlines.append(builGroupLinesByMonth (1,  SuperMaps, "TOTAL VOTES TREND", typeFreq));
				   Superlines.append(builGroupLinesByMonth (2,  SuperMaps, "NUM REVIEWS TREND", typeFreq));
				   return Superlines;
		   default:
			   return null;
		    }

		   
				   
	   }
	   
	   private static StringBuffer builGroupLinesByYear (int index, HashMap<String, TreeMap<Float, Float>>[] SuperMaps, String label, freqType typeFreq) throws Exception { 

		   
		   StringBuffer Superlines = new StringBuffer();
		   StringBuffer headerLine = buildHeader(listType.by_Prod2, typeFreq, label);
		   StringBuffer lines = new StringBuffer();
		   for (String prodID: SuperMaps[index].keySet()) {
			   TreeMap<Float, Float> tree = SuperMaps[index].get(prodID);

			   if (tree == null || (!tree.containsKey(Float.parseFloat("2013.0")) || !tree.containsKey(Float.parseFloat("2014.0")) 
					   || !tree.containsKey(Float.parseFloat("2015.0")) || !tree.containsKey(Float.parseFloat("2016.0")) ))
						   continue;

			   if (tree.size() > 0)
				   lines.append(prodID).append(";");
			   for (float year = (float)startYear; year <= (float)endYear; year++)	 {
				   if(!tree.containsKey(year)){
					   lines.append(0).append(";");
				   } else
					   lines.append(String.format("%,4.2f",tree.get(year))).append(";");
			   }
			   lines.append("\n");
		   }
		   Superlines.append(headerLine).append(lines);
		   return Superlines;
  
		   
		   
	   }

	   
	   private static StringBuffer builGroupLinesByMonth (int index, HashMap<String, TreeMap<Float, Float>>[] SuperMaps, String label, freqType typeFreq) throws Exception { 

		   
		   StringBuffer Superlines = new StringBuffer();
		   StringBuffer headerLine = buildHeader(listType.by_Prod2, typeFreq, label);
		   StringBuffer lines = new StringBuffer();
		   for (String prodID: SuperMaps[index].keySet()) {
			   TreeMap<Float, Float> tree = SuperMaps[index].get(prodID);
//			   if (tree == null || (!tree.containsKey(Float.parseFloat("2013.01")) || !tree.containsKey(Float.parseFloat("2014.01")) || !tree.containsKey(Float.parseFloat("2015.01")) || !tree.containsKey(Float.parseFloat("2016.01")) ))
//						   continue;
			   if (tree == null || tree.size() < minNumbervaluesbyMonth)
					   continue;
			   if (tree.size() > 0)
				   lines.append(prodID).append(";");
			   float key = 0F;
			   for (float year = (float)startYear; year <= (float)endYear; year++)	 {
					for (int i=1; i <= 12; i++) {
						   key = year+((float)i/100);
						   if(!tree.containsKey(key)){
							   lines.append(0).append(";");
						   } else
							   lines.append(String.format("%,4.2f",tree.get(key))).append(";");

					}
				}

			   lines.append("\n");
		   }
		   
		   
		   Superlines.append(headerLine).append(lines);
		   return Superlines;
		   
	   }

	   



	   private static StringBuffer buildHeader(listType typeList, freqType typeFreq, String typeChart) throws Exception { 
	   
		   StringBuffer headerLine = new StringBuffer();
		   switch (typeList) { 	
			   case by_Prod:
			   case all: 
					headerLine.append("PROD ID").append(";").append("YEAR").append(";").append("RATING (mean)").append(";").append("RATING (sd)").append(";").append("SAMPLE ENT (mean)").append(";")
					.append("POS VOTES (mean)").append(";").append("POS VOTES (sd)").append(";").append("TOTAL VOTES (mean)").append(";").append("TOTAL VOTES (sd)").append(";").append("LEN REVIEW (mean)").append(";")
					.append("LEN REVIEW (sd)").append(";").append("# REVIEWS").append("\n");
					break;
			   case by_Prod2:				   
				    switch (typeFreq) { 
				    
					   	case year:  
							headerLine.append("\n").append(typeChart).append("\n").append("PROD ID").append(";").append("2012").append(";").append("2013").append(";").append("2014").append(";").append("2015").append(";")
							.append("2016").append(";").append("2017").append("\n");
							break;

					   	case month:  
							headerLine.append("\n").append(typeChart).append("\n").append("PROD ID").append(";").append("2012 JAN").append(";").append("2012 FEB").append(";").append("2012 MAR").append(";").append("2012 APR").append(";").append("2012 MAY").append(";").append("2012 JUN").append(";")
							.append("2012 JUL").append(";").append("2012 AUG").append(";").append("2012 SEP").append(";").append("2012 OCT").append(";").append("2012 NOV").append(";").append("2012 DEC").append(";").append("2013 JAN").append(";");
							for (int j=0; j < 5; j++) {
								headerLine.append(String.valueOf(2013+j));
								for (int i=0; i < 12; i++)
									headerLine.append(";");
							}
							headerLine.append("\n");
							break;
				    }
				   
		   }
		   return headerLine;
		
	   }

	   private static void loadRevPerYear ()  throws Exception { 

		   for (Review r: patterns) { 
			   
				Vector<Review> ReviewPerY;
				if (ReviewPerYear.get(Float.parseFloat(CheckData.getYearFromCalendar(r.getData()))) == null){ 
					ReviewPerY = new Vector<Review>();
				} else { 
					ReviewPerY = ReviewPerYear.get(Float.parseFloat(CheckData.getYearFromCalendar(r.getData())));
				}
				ReviewPerY.add(r);
				ReviewPerYear.put(Float.parseFloat(CheckData.getYearFromCalendar(r.getData())), ReviewPerY);

		   }

	   }
	   
	   /**
	    * launch loadRevPerYear () ->load data for each year, and after loadRevPerMonth () -> load data for each year.month
	    * @throws Exception
	    */
	   private static void loadRevPerMonth ()  throws Exception { 

		   	   @SuppressWarnings("unused")
			   TreeMap<Float, Vector<Review>> ReviewPerYearClone =  (TreeMap<Float, Vector<Review>>) ReviewPerYear.clone();
		   	   TreeMap<Integer, Vector<Review>> ReviewPerMonth = new TreeMap<Integer, Vector<Review>>() ;			   
			   
			   for (Float year: ReviewPerYearClone.keySet()) 	{
				   
				   ReviewPerMonth = new TreeMap<Integer, Vector<Review>>() ;		
				   for (Review r: patterns) {

					   if (Integer.parseInt(CheckData.getYearFromCalendar(r.getData()) ) == year) { 
						   
							Vector<Review> ReviewPerM;
							if (ReviewPerMonth.get(Integer.parseInt(CheckData.getMonthFromCalendar(r.getData()))) == null){ 
								ReviewPerM = new Vector<Review>();
							} else { 
								ReviewPerM = ReviewPerMonth.get(Integer.parseInt(CheckData.getMonthFromCalendar(r.getData())));
							}
							ReviewPerM.add(r);
							ReviewPerMonth.put(Integer.parseInt(CheckData.getMonthFromCalendar(r.getData())), ReviewPerM);
						   
					   }
	
				   }
				   for (Integer month: ReviewPerMonth.keySet()) 	{
					   ReviewPerYear.remove(year);
					   float key = year+((float)month/100);
					   ReviewPerYear.put(key,ReviewPerMonth.get(month));		// es.: ReviewPerYear (year.001, value of january of year)
					   System.out.println(key+"->"+ReviewPerMonth.get(month).size());
				   }
			   }

	   }

	   private static  TreeMap<Float, Float>[] calculateMeans(List<String> dataLines, String prodID) throws Exception { 
		   
		    @SuppressWarnings("unchecked")
		    TreeMap<Float, Float>[] Maps = new TreeMap[3];
		    TreeMap<Float, Float> MapPositVotes = new TreeMap<Float, Float>();
		    TreeMap<Float, Float> MapTotalVotes = new TreeMap<Float, Float>();
		    TreeMap<Float, Float> MapNumReviews = new TreeMap<Float, Float>();
		   
			for (Float year: ReviewPerYear.keySet()) 	{
				
				   StringBuffer dataLine = new StringBuffer();	
				   Vector<Review>  ReviewPerYorM = ReviewPerYear.get(year);
				   Vector<Double>  entropies = new Vector<Double>();
				   Vector<Integer> positVotes = new Vector<Integer>();
				   Vector<Integer> totalVotes = new Vector<Integer>();
				   Vector<Integer> ratings = new Vector<Integer>();
				   Vector<Integer> lenReviews = new Vector<Integer>();
				   		
				   int numReviews = 0;
				   for (Review r: ReviewPerYorM) {
					   if (r.getProdID().equals(prodID) || prodID.isEmpty()) {
						   entropies.add(r.getSampEnt());
						   positVotes.add(r.getHelpfulVote());
						   totalVotes.add(r.getTotalVote());
						   ratings.add(r.getRating());
						   lenReviews.add(r.getTextLength());
						   numReviews++;
					   }
				   }
				   
				   if (numReviews > 0) {
					   Double meanEnt = Mathemat.meanDouble(entropies);
					   Float meanPosVotes = Mathemat.meanInt(positVotes);
					   Float sdPosVotes = Mathemat.sdInt(positVotes);				   
					   Float meanTotVotes = Mathemat.meanInt(totalVotes);
					   Float sdTotVotes = Mathemat.sdInt(totalVotes);				   
					   Float meanRating = Mathemat.meanInt(ratings);
					   Float sdRating = Mathemat.sdInt(ratings);
					   Float meanLenText = Mathemat.meanInt(lenReviews);
					   Float sdLenText = Mathemat.sdInt(lenReviews);	
					   
					   
					   if (year >= startYear && meanPosVotes > 0)  {
						   MapPositVotes.put(year, meanPosVotes);
						   MapTotalVotes.put(year, meanTotVotes);
						   MapNumReviews.put(year, Float.valueOf(numReviews));						   
					   }
					
				  
						dataLine.append(prodID).append(";").append(year).append(";").append(String.format("%,4.2f",meanRating)).append(";").append(String.format("%,4.2f",sdRating)).append(";").append(String.format("%,4.2f",meanEnt)).append(";")
						.append(String.format("%,4.2f",meanPosVotes)).append(";").append(String.format("%,4.2f",sdPosVotes)).append(";").append(String.format("%,4.2f",meanTotVotes)).append(";").append(String.format("%,4.2f",sdTotVotes)).append(";")
						.append(String.format("%,4.2f",meanLenText)).append(";").append(String.format("%,4.2f",sdLenText)).append(";").append(numReviews).append("\n");
					
						dataLines.add(dataLine.toString());
				   }
			}
			
			
			Maps[0] = MapPositVotes;
			Maps[1] = MapTotalVotes;
			Maps[2] = MapNumReviews;
			
			return Maps;
			
		
		
	   }


	   
	   private static Review splitFieldsReviews(String line) throws Exception { 
	    	
	    	String fields[] = line.split(";");
	    	
	    	Integer reviewID = Integer.parseInt(fields[0]);
	    	Integer rating = Integer.parseInt(fields[2]);
	       	SimpleDateFormat  format = new SimpleDateFormat("yyyy-MM-dd");  
	    	Date date = format.parse(fields[1].trim());
	    	Double sampEnt = Double.parseDouble(fields[3].trim().replaceAll("\\,", "."));
	    	Integer helpfulVote = (int) (Double.parseDouble(fields[7].trim().replaceAll("\\,", ".")));
	    	Integer totalVote = (int) (Double.parseDouble(fields[8].trim().replaceAll("\\,", ".")));
	    	Double  helpfulVotePerCent = Double.parseDouble(fields[6].trim().replaceAll("\\,", "."));
	    	Integer reviewsXprod = (int) (Double.parseDouble(fields[9].trim().replaceAll("\\,", ".")));
	    	String  prodID = fields[10];
	       	Integer textLengthNormal= Integer.parseInt(fields[11]);        
	    	String textNormal = fields[12];
	    	
	     	return new Review( reviewID,   rating,   textNormal,  textLengthNormal, 0,  date,  helpfulVote,  totalVote, helpfulVotePerCent, prodID, sampEnt, reviewsXprod);
	     	
	     	
	    }

	   // USEFUL CheckData.getYearFromCalendar(Date date) 

}
