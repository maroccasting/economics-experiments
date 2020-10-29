package massive.clearsight.Vote_Rate_Entropy;

import massive.clearsight.Vote_Rate_Entropy.Reviews.distType;
import massive.clearsight.tools.semantics.utils.Metrics;

public class SampEntropy {

	public SampEntropy() {
		
	}
	

	/**
	 * @param dim embedded dimension
	 * @param r	tolerance (typically 0.2 * std)
	 * @param data	time-series data
	 * @return entropy
	 * @throws Exception
	 */
	public static double entropy ( int dim, int diffDim, double r, Double data[], distType typeDist) throws Exception {
		
		double dist = 0.0;
		int A = 0;
		int B = 0;

		for (int m = dim; m <= dim+diffDim; m+=diffDim) {
			
			Double[] dataMat1 = new Double[m];
			Double[] dataMat2 = new Double[m];
			for (int i = 0; i < data.length - m; i++) {
				dataMat1 = partArray(data, i, m);
				for (int j = 0; j < data.length - m; j++) {
					dataMat2 = partArray(data, j, m);
					if (i != j) {
						dist = getDistance(dataMat1, dataMat2, typeDist);
						if (dist <= r) {
							if (m == dim) 
								B++;
							else if (m == (dim+diffDim))
								A++;
						}

					}
			
				}
			}
		
		}
		if (A == 0 || B == 0)
			return 0;
		else
			return - Math.log((double)A/B);
		// if A/B >> 1 then there is dispersion (if the entropy is high in absolute value then the system is more unpredictable) -> less useful 
	}
	
	
	public static double getDistance(Double[] features1, Double[] features2, distType typeDist) { 
		
		switch (typeDist) {  // distType {ChebyshevDistance, EuclidDistance};			
			 case ChebyshevDistance:  return getChebyshevDistance(features1, features2);
			 
			 case EuclidDistance:  return  Metrics.euclidDistance(features1, features2);
			 
			 default:  return  Metrics.euclidDistance(features1, features2);
				
		}
		
		
	}
	
	   /**
     * Chebyshev Distance implementation. 
     * Both features must have the same length 
     * @param features1 first vector to compare 
     * @param features2 second vector to compare 
     * @return Chebyshev distance between two feature vectors 
     */ 
    public static double getChebyshevDistance(Double[] features1, Double[] features2) { 
    	
    	double distance = 0.0; 
    	if(features1 == null || features2 == null) {
    		throw new IllegalArgumentException("One or both features are null ");
    	}
        if(features1.length != features2.length) { 
            throw new IllegalArgumentException("Both features should have the same length. Received lengths of [" + 
                    + features1.length + "] and [" + features2.length + "]"); 
        } 
        
        for (int i = 0; i < features1.length; i++) { 
            double currentDistance = Math.abs(features1[i] - features2[i]); 
            distance = (currentDistance > distance) ? currentDistance : distance;  
        } 
        return distance; 
    } 

    /**
    * Nullity check of parameters
    * @param features1 the first features vector
    * @param features2 the secund features vector
    * @return Double.POSITIVE_INFINITY in case either or both of the vectors are null, a negative number otherwise
    */
   protected static double positiveInfinityIfEitherOrBothAreNull(Double[] features1, Double[] features2) {
       if(features1 == null || features2 == null) {
           return Double.POSITIVE_INFINITY;
       } else {
           return -1.0d;
       }
   }
	/**
	 * 
	 * @param array	source array
	 * @param pos	source and dest pos
	 * @param size	dim of data to copy
	 * @return the array copied
	 */
	public static Double[] partArray(Double[] array, int pos, int size) {
		if (pos + size > array.length)
			return null;
		
		Double[] part = new Double[size];	    
	    for (int i = 0; i < part.length; i++) {
	        part[i] = array[i+pos];
	    }
	    return part;
	}

}
