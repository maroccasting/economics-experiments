package massive.clearsight.Vote_Rate_Entropy;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Date;

import massive.clearsight.Vote_Rate_Entropy.Reviews.Sentiment;



public class Review implements Comparable<Review>, Serializable{

	private static final long serialVersionUID = 5589694393443638909L;

	
	private Integer ReviewID;
	private String  Title;
	private String  TitleNormal;
	private Integer Rating;
	private String  Text;
	private String  TextUltimate;
	private Integer TextLength;
	private Integer TextLengthNormal;
	private String  TextNormal;
	private String  POS;
	private Date Data;
	private String  URL;
	private Integer  HelpfulVote;
	private Double  HelpfulVotePerCent;
	private Integer  TotalVote;
	private Integer  NumInfoUseful;
	private  Double RatioUsefullAll;	// NEW
	private ArrayList<WordWeight>  DataUseful;		// NEW
	private Double Dens;
	private String  PROS;
	private String  CONS;
	private String ProdID;
	private String ReviewerID;
	private Double SampEnt;
	private Double ApproxEnt;
	private Double SpecialRelative;
	private int Positive;
	private int Negative;
	private Double RatioUsefullAllHalf ;
	private double RatioLTitleLText;
	private Sentiment Senti ;
	private Sentiment SentiHalf ;
	private Double ari; 


	// add density = #ones/#all (ones and zeros)
	// density * (1 - entropy)

	

	public Review(Integer reviewID,  Integer rating,  String textNormal, Integer textLengthNormal, Integer textLength, Date data, Integer helpfulVote, Integer totalVote, Double  helpfulVotePerCent, String prodID, Double sampEnt, 
			Integer reviewsXprod) {
		super();
		ReviewID = reviewID;
		Rating = rating;
		TextLengthNormal = textLengthNormal;
		TextLength = textLength;
		TextNormal = textNormal;
		Data = data;
		HelpfulVote = helpfulVote;
		TotalVote = totalVote;
		HelpfulVotePerCent = helpfulVotePerCent;
		ProdID = prodID;
		SampEnt = sampEnt;
		Reviews.numReviewsForProd.put(prodID, reviewsXprod);
	}

	public Review(Integer reviewID, String title, Integer rating, String titleNormal, String text, String textNormal, Integer textLengthNormal, Integer textLength,  String pOS,
			Date data, String uRL, Integer helpfulVote, Integer totalVote, Double  helpfulVotePerCent, String prodID, String reviewerID) {
		super();
		ReviewID = reviewID;
		Title = title;
		Rating = rating;
		TitleNormal = titleNormal;
		Text = text;
		TextLengthNormal = textLengthNormal;
		TextLength = textLength;
		TextNormal = textNormal;
		POS = pOS;
		Data = data;
		URL = uRL;
		HelpfulVote = helpfulVote;
		TotalVote = totalVote;
		HelpfulVotePerCent = helpfulVotePerCent;
		ProdID = prodID;
		ReviewerID = reviewerID;
	}

	


	public Review(String reviewerID, Integer reviewID, String title, Integer rating, String text, String  textUltimate, Integer textLengthNormal, Integer textLength,  Date data, String uRL, 
			Double  helpfulVotePerCent, Integer helpfulVote, Integer totalVote, Integer numInfoUseful, Double ratioUsefullAll, Double ratioUsefullAllHalf , ArrayList<WordWeight> dataUseful,
			String prodID, Double sampEnt, Double approxEnt, Double dens, Double specialRelative, int positive, int negative, double ratioLTitleLText, Sentiment senti , Sentiment sentiHalf, Double ARI ) {
		
		super();
		ReviewID = reviewID;
		Title = title;
		Rating = rating;
		Text = text;
		TextUltimate = textUltimate;
		TextLengthNormal = textLengthNormal;
		TextLength = textLength;
		Data = data;
		URL = uRL;
		HelpfulVotePerCent = helpfulVotePerCent;
		HelpfulVote = helpfulVote;
		RatioUsefullAll = ratioUsefullAll; 
		DataUseful = dataUseful;
		TotalVote = totalVote;
		NumInfoUseful = numInfoUseful;
		ProdID = prodID;
		SampEnt = sampEnt;
		ApproxEnt = approxEnt;
		Dens = dens;
		ReviewerID = reviewerID;
		SpecialRelative = specialRelative;		
		RatioUsefullAllHalf =ratioUsefullAllHalf;
		RatioLTitleLText =ratioLTitleLText;
		Senti = senti;
		SentiHalf = sentiHalf;
		Positive = positive;
		Negative = negative;
		ari = ARI;

	}



	public Review(Integer reviewID, String title, String titleNormal, Integer rating, String text, String textNormal,
			String pOS, Date data, String uRL, Integer helpfulVote, Integer totalVote, Double  helpfulVotePerCent, Integer numInfoUseful,  
			String pROS, String cONS, String prodID, String reviewerID, Double sampEnt, Double approxEnt) {
		super();
		ReviewID = reviewID;
		Title = title;
		TitleNormal = titleNormal;
		Rating = rating;
		Text = text;
		TextNormal = textNormal;
		POS = pOS;
		Data = data;
		URL = uRL;
		HelpfulVote = helpfulVote;
		TotalVote = totalVote;
		HelpfulVotePerCent = helpfulVotePerCent;
		NumInfoUseful = numInfoUseful;
		PROS = pROS;
		CONS = cONS;
		ProdID = prodID;
		ReviewerID = reviewerID;
		SampEnt = sampEnt;
		ApproxEnt = approxEnt;
	}



	public Double getRatioUsefullAllHalf() {
		return RatioUsefullAllHalf;
	}

	public void setRatioUsefullAllHalf(Double ratioUsefullAllHalf) {
		RatioUsefullAllHalf = ratioUsefullAllHalf;
	}

	public double getRatioLTitleLText() {
		return RatioLTitleLText;
	}

	public void setRatioLTitleLText(double ratioLTitleLText) {
		RatioLTitleLText = ratioLTitleLText;
	}

	public Sentiment getSenti() {
		return Senti;
	}

	public void setSenti(Sentiment senti) {
		Senti = senti;
	}

	public Sentiment getSentiHalf() {
		return SentiHalf;
	}

	public void setSentiHalf(Sentiment sentiHalf) {
		SentiHalf = sentiHalf;
	}

	public Integer getReviewID() {
		return ReviewID;
	}


	public void setReviewID(Integer reviewID) {
		ReviewID = reviewID;
	}


	public String getTitle() {
		return Title;
	}


	public void setTitle(String title) {
		Title = title;
	}


	public String getTitleNormal() {
		return TitleNormal;
	}


	public void setTitleNormal(String titleNormal) {
		TitleNormal = titleNormal;
	}


	public String getText() {
		return Text;
	}


	public void setText(String text) {
		Text = text;
	}


	public String getTextNormal() {
		return TextNormal;
	}


	public void setTextNormal(String textNormal) {
		TextNormal = textNormal;
	}


	public String getPOS() {
		return POS;
	}


	public void setPOS(String pOS) {
		POS = pOS;
	}


	public Date getData() {
		return Data;
	}


	public void setData(Date data) {
		Data = data;
	}


	public String getURL() {
		return URL;
	}


	public void setURL(String uRL) {
		URL = uRL;
	}


	public Integer getHelpfulVote() {
		return HelpfulVote;
	}


	public void setHelpfulVote(Integer helpfulVote) {
		HelpfulVote = helpfulVote;
	}


	public Integer getTotalVote() {
		return TotalVote;
	}


	public void setTotalVote(Integer totalVote) {
		TotalVote = totalVote;
	}


	public String getPROS() {
		return PROS;
	}


	public void setPROS(String pROS) {
		PROS = pROS;
	}


	public String getCONS() {
		return CONS;
	}


	public void setCONS(String cONS) {
		CONS = cONS;
	}


	public String getProdID() {
		return ProdID;
	}


	public void setProdID(String prodID) {
		ProdID = prodID;
	}


	public String getReviewerID() {
		return ReviewerID;
	}


	public void setReviewerID(String reviewerID) {
		ReviewerID = reviewerID;
	}

	

	public Integer getRating() {
		return Rating;
	}



	public void setRating(Integer rating) {
		Rating = rating;
	}



	public Double getSampEnt() {
		return SampEnt;
	}



	public void setSampEnt(Double sampEnt) {
		SampEnt = sampEnt;
	}



	public Double getApproxEnt() {
		return ApproxEnt;
	}



	public void setApproxEnt(Double approxEnt) {
		ApproxEnt = approxEnt;
	}

	

	public Integer getTextLengthNormal() {
		return TextLengthNormal;
	}



	public void setTextLengthNormal(Integer TextLengthNormal) {
		this.TextLengthNormal = TextLengthNormal;
	}



	public Integer getTextLength() {
		return TextLength;
	}

	public void setTextLength(Integer textLength) {
		TextLength = textLength;
	}

	public String getTextUltimate() {
		return TextUltimate;
	}



	public void setTextUltimate(String textUltimate) {
		TextUltimate = textUltimate;
	}



	public Double getHelpfulVotePerCent() {
		return HelpfulVotePerCent;
	}



	public void setHelpfulVotePerCent(Double helpfulVotePerCent) {
		HelpfulVotePerCent = helpfulVotePerCent;
	}

	


	public Integer getNumInfoUseful() {
		return NumInfoUseful;
	}



	public void setNumInfoUseful(Integer numInfoUseful) {
		NumInfoUseful = numInfoUseful;
	}

	




	public Double getRatioUsefullAll() {
		return RatioUsefullAll;
	}




	public void setRatioUsefullAll(Double ratioUsefullAll) {
		RatioUsefullAll = ratioUsefullAll;
	}




	public ArrayList<WordWeight>  getDataUseful() {
		return DataUseful;
	}




	public void setDataUseful(ArrayList<WordWeight> dataUseful) {
		DataUseful = dataUseful;
	}


	public Double getDens() {
		return Dens;
	}

	public void setDen(Double Dens) {
		this.Dens = Dens;
	}

	

	
	public Double getSpecialRelative() {
		return SpecialRelative;
	}

	public void setSpecialRelative(Double specialRelative) {
		SpecialRelative = specialRelative;
	}
	
	

	public int getPositive() {
		return Positive;
	}

	public void setPositive(int positive) {
		Positive = positive;
	}

	public int getNegative() {
		return Negative;
	}

	public void setNegative(int negative) {
		Negative = negative;
	}
	
	

	public Double getAri() {
		return ari;
	}

	public void setAri(Double ari) {
		this.ari = ari;
	}

	@Override
	public int compareTo(Review o) {
		
		 
		 switch (Reviews.typeOrder) {
		 
		 
			 case Rating:   
				 return (Reviews.inv) ? - Rating.compareTo(o.Rating) : Rating.compareTo(o.Rating);
			 
			 case TextLenghtNormal:   
				 return (Reviews.inv) ? - TextLengthNormal.compareTo(o.TextLengthNormal) : TextLengthNormal.compareTo(o.TextLengthNormal);
		 
			 case Date:  
				 return (Reviews.inv) ? - Data.compareTo(o.Data) : Data.compareTo(o.Data);
				 
			 case HelpfulVote:  
				 return (Reviews.inv) ? - HelpfulVote.compareTo(o.HelpfulVote) : HelpfulVote.compareTo(o.HelpfulVote);

			 case TotalVote:  
				 return (Reviews.inv) ? - TotalVote.compareTo(o.TotalVote) : TotalVote.compareTo(o.TotalVote);
				 
			 case Dens:  
				 return (Reviews.inv) ? - Dens.compareTo(o.Dens) : Dens.compareTo(o.Dens);
				 
			 case PercentInfoUseful:  
				 return (Reviews.inv) ? - RatioUsefullAll.compareTo(o.RatioUsefullAll) : RatioUsefullAll.compareTo(o.RatioUsefullAll);

			 case SampEnt:  
				 return (Reviews.inv) ? - SampEnt.compareTo(o.SampEnt) : SampEnt.compareTo(o.SampEnt);

			 case ApproxEnt:  
				 return (Reviews.inv) ? - ApproxEnt.compareTo(o.ApproxEnt) : ApproxEnt.compareTo(o.ApproxEnt);
				 
			case  ProdDateChain:            if (ProdID.compareTo(o.ProdID) == 0) { 
								            	return (Reviews.inv) ? - Data.compareTo(o.Data) : Data.compareTo(o.Data); 
								            /* If they are not equal, we just order by the primary elements */
								            } else {
								                return ProdID.compareTo(o.ProdID);
								            }
				 
			 case NumReviewsForProd:  
				 return (Reviews.inv) ? - Reviews.numReviewsForProd.get(ProdID).compareTo(Reviews.numReviewsForProd.get(o.ProdID)) : 
					 					  Reviews.numReviewsForProd.get(ProdID).compareTo(Reviews.numReviewsForProd.get(o.ProdID));
				 
			 default:  return (Reviews.inv) ? - Rating.compareTo(o.Rating) : Rating.compareTo(o.Rating);

		 }
		
         
	}


	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((ApproxEnt == null) ? 0 : ApproxEnt.hashCode());
		result = prime * result + ((CONS == null) ? 0 : CONS.hashCode());
		result = prime * result + ((Data == null) ? 0 : Data.hashCode());
		result = prime * result + ((HelpfulVote == null) ? 0 : HelpfulVote.hashCode());
		result = prime * result + ((HelpfulVotePerCent == null) ? 0 : HelpfulVotePerCent.hashCode());
		result = prime * result + ((NumInfoUseful == null) ? 0 : NumInfoUseful.hashCode());
		result = prime * result + ((POS == null) ? 0 : POS.hashCode());
		result = prime * result + ((PROS == null) ? 0 : PROS.hashCode());
		result = prime * result + ((ProdID == null) ? 0 : ProdID.hashCode());
		result = prime * result + ((Rating == null) ? 0 : Rating.hashCode());
		result = prime * result + ((ReviewID == null) ? 0 : ReviewID.hashCode());
		result = prime * result + ((ReviewerID == null) ? 0 : ReviewerID.hashCode());
		result = prime * result + ((SampEnt == null) ? 0 : SampEnt.hashCode());
		result = prime * result + ((Text == null) ? 0 : Text.hashCode());
		result = prime * result + ((TextLengthNormal == null) ? 0 : TextLengthNormal.hashCode());
		result = prime * result + ((TextNormal == null) ? 0 : TextNormal.hashCode());
		result = prime * result + ((Title == null) ? 0 : Title.hashCode());
		result = prime * result + ((TitleNormal == null) ? 0 : TitleNormal.hashCode());
		result = prime * result + ((TotalVote == null) ? 0 : TotalVote.hashCode());
		result = prime * result + ((URL == null) ? 0 : URL.hashCode());
		return result;
	}



	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Review other = (Review) obj;
		if (ApproxEnt == null) {
			if (other.ApproxEnt != null)
				return false;
		} else if (!ApproxEnt.equals(other.ApproxEnt))
			return false;
		if (CONS == null) {
			if (other.CONS != null)
				return false;
		} else if (!CONS.equals(other.CONS))
			return false;
		if (Data == null) {
			if (other.Data != null)
				return false;
		} else if (!Data.equals(other.Data))
			return false;
		if (HelpfulVote == null) {
			if (other.HelpfulVote != null)
				return false;
		} else if (!HelpfulVote.equals(other.HelpfulVote))
			return false;
		if (HelpfulVotePerCent == null) {
			if (other.HelpfulVotePerCent != null)
				return false;
		} else if (!HelpfulVotePerCent.equals(other.HelpfulVotePerCent))
			return false;
		if (NumInfoUseful == null) {
			if (other.NumInfoUseful != null)
				return false;
		} else if (!NumInfoUseful.equals(other.NumInfoUseful))
			return false;
		if (POS == null) {
			if (other.POS != null)
				return false;
		} else if (!POS.equals(other.POS))
			return false;
		if (PROS == null) {
			if (other.PROS != null)
				return false;
		} else if (!PROS.equals(other.PROS))
			return false;
		if (ProdID == null) {
			if (other.ProdID != null)
				return false;
		} else if (!ProdID.equals(other.ProdID))
			return false;
		if (Rating == null) {
			if (other.Rating != null)
				return false;
		} else if (!Rating.equals(other.Rating))
			return false;
		if (ReviewID == null) {
			if (other.ReviewID != null)
				return false;
		} else if (!ReviewID.equals(other.ReviewID))
			return false;
		if (ReviewerID == null) {
			if (other.ReviewerID != null)
				return false;
		} else if (!ReviewerID.equals(other.ReviewerID))
			return false;
		if (SampEnt == null) {
			if (other.SampEnt != null)
				return false;
		} else if (!SampEnt.equals(other.SampEnt))
			return false;
		if (Text == null) {
			if (other.Text != null)
				return false;
		} else if (!Text.equals(other.Text))
			return false;
		if (TextLengthNormal == null) {
			if (other.TextLengthNormal != null)
				return false;
		} else if (!TextLengthNormal.equals(other.TextLengthNormal))
			return false;
		if (TextNormal == null) {
			if (other.TextNormal != null)
				return false;
		} else if (!TextNormal.equals(other.TextNormal))
			return false;
		if (Title == null) {
			if (other.Title != null)
				return false;
		} else if (!Title.equals(other.Title))
			return false;
		if (TitleNormal == null) {
			if (other.TitleNormal != null)
				return false;
		} else if (!TitleNormal.equals(other.TitleNormal))
			return false;
		if (TotalVote == null) {
			if (other.TotalVote != null)
				return false;
		} else if (!TotalVote.equals(other.TotalVote))
			return false;
		if (URL == null) {
			if (other.URL != null)
				return false;
		} else if (!URL.equals(other.URL))
			return false;
		return true;
	}



	@Override
	public String toString() {
		return "Review [ReviewID=" + ReviewID + ",  TextLength " + TextLength +", Rating="
				+ Rating + ",  Data=" + Data + ", HelpfulVotePerCent=" + HelpfulVotePerCent + ", HelpfulVote=" + HelpfulVote + ", TotalVote=" + TotalVote + 
				", RatioUsefullAll=" + RatioUsefullAll + ", ProdID=" + ProdID + ", SampEnt=" + SampEnt + ", ApproxEnt=" + ApproxEnt + ", "
						+ "Title=" + Title + ", TextUltimate= " + TextUltimate +"]";
	}



//	public String toStringVerbose() {
//		return "Review [ReviewID=" + ReviewID + ", Title=" + Title + ", TitleNormal=" + TitleNormal + ", Rating="
//				+ Rating + ", TextLength " + TextLength +", Text=" + Text + ", TextNormal=" + TextNormal + ", POS=" + POS + ", Data=" + Data
//				+ ", URL=" + URL + ", HelpfulVote=" + HelpfulVote + ", TotalVote=" + TotalVote + ", PROS=" + PROS
//				+ ", CONS=" + CONS + ", ProdID=" + ProdID + ", ReviewerID=" + ReviewerID + ", SampEnt=" + SampEnt
//				+ ", ApproxEnt=" + ApproxEnt + "]";
//	}




	
	public String toStringVerbose() {
		return "Review [ReviewID=" + ReviewID + ", Title=" + Title + ", TitleNormal=" + TitleNormal + ", Rating="
				+ Rating + ", Text=" + Text + ", TextUltimate=" + TextUltimate + ", TextLength=" + TextLength
				+ ", TextNormal=" + TextNormal + ", POS=" + POS + ", Data=" + Data + ", URL=" + URL + ", HelpfulVote="
				+ HelpfulVote + ", HelpfulVotePerCent=" + HelpfulVotePerCent + ", TotalVote=" + TotalVote + ", NumInfoUseful=" + NumInfoUseful
				+ ", RatioUsefullAll=" + RatioUsefullAll + ", DataUseful=" + DataUseful
				+ ", PROS=" + PROS + ", CONS=" + CONS + ", ProdID="
				+ ProdID + ", ReviewerID=" + ReviewerID + ", SampEnt=" + SampEnt + ", ApproxEnt=" + ApproxEnt + "]";
	}



	
	
	
	

}
