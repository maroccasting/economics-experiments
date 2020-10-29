package massive.clearsight.Vote_Rate_Entropy;

import massive.clearsight.Vote_Rate_Entropy.Reviews.distType;

public class ApproxEntropy {

	public ApproxEntropy() {
		
	}
	

	/**
	 * @param dim embedded dimension
	 * @param r	tolerance (typically 0.2 * std)
	 * @param data	time-series data
	 * @return entropy
	 * @throws Exception
	 */
	public static double entropy ( int dim, double r, Double data[], distType typeDist) throws Exception {
		
			double[] C = new double[data.length - dim +1];
			double[] C_tot = new double[2];
			double dist = 0.0;
			int n = data.length ;

			for (int m = dim; m <= dim+1; m++) {
				
				Double[] dataMat1 = new Double[m];
				Double[] dataMat2 = new Double[m];
				C = new double[data.length - dim +1];
				
				for (int i = 0; i < data.length - m + 1; i++) {
					dataMat1 = SampEntropy.partArray(data, i, m);
					for (int j = 0; j < data.length - m + 1; j++) {
						dataMat2 = SampEntropy.partArray(data, j, m);
						if (i != j) {
							dist = SampEntropy.getDistance(dataMat1, dataMat2, typeDist);
							if (dist <= r) {
								C[i]++;
							}
	
						}
				
					}
					C[i] = C[i]/(n - m );
					C_tot[m - dim] += (C[i] != 0) ? Math.log((double)C[i]) : 0;
				}
				C_tot[m - dim] = C_tot[m - dim] / (n - m);
			
			}

			return (C_tot[0] - C_tot[1]);  // verify! strange that vectors of dim get a major value than vectors of dim+1 
		
	}
	


}
