package massive.clearsight.Vote_Rate_Entropy;


public class WordWeight {
	
	private int tf;
	private double logTf;
//	private double IDF;
//	private double probIDF;
	private double tfIDF;
	private double probtfIDF;
	private String word;


	public WordWeight() {
		// TODO Auto-generated constructor stub
	}
	



	public WordWeight(String word, int tf, double logTf, double tfIDF, double probtfIDF) {
		super();
		this.tf = tf;
		this.logTf = logTf;
		this.tfIDF = tfIDF;
		this.probtfIDF = probtfIDF;
		this.word = word;
	}


//	public WordWeight(int tf, double logTf, double iDF, double probIDF, double tfIDF, double probtfIDF) {
//		super();
//		this.tf = tf;
//		this.logTf = logTf;
//		IDF = iDF;
//		this.probIDF = probIDF;
//		this.tfIDF = tfIDF;
//		this.probtfIDF = probtfIDF;
//	}




	public int getTf() {
		return tf;
	}


	public void setTf(int tf) {
		this.tf = tf;
	}


	public double getLogTf() {
		return logTf;
	}


	public void setLogTf(double logTf) {
		this.logTf = logTf;
	}


//	public double getIDF() {
//		return IDF;
//	}
//
//
//	public void setIDF(double iDF) {
//		IDF = iDF;
//	}
//
//
//	public double getProbIDF() {
//		return probIDF;
//	}
//
//
//	public void setProbIDF(double probIDF) {
//		this.probIDF = probIDF;
//	}


	public double getTfIDF() {
		return tfIDF;
	}


	public void setTfIDF(double tfIDF) {
		this.tfIDF = tfIDF;
	}

	

	public double getProbtfIDF() {
		return probtfIDF;
	}


	public void setProbtfIDF(double probtfIDF) {
		this.probtfIDF = probtfIDF;
	}


	@Override
	public String toString() {
		return "WordWeight [word="+word+", tf=" + tf + ", logTf=" + logTf + ", tfIDF="
				+ tfIDF + ", probtfIDF=" + probtfIDF + "]";
	}



	
	
	
}
