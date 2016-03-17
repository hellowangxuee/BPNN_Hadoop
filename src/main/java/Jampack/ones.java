package Jampack;

/**
   Eye generates a matrix whose diagonal elements are one
   and whose off diagonal elements are zero.
*/

public class ones{
   
   public static Zmat o(int m, int n){

      Zmat I = new Zmat(m, n);

      for (int i=0;i<m;i++){
    	  for (int j=0;j<n;j++){
    		  I.re[i][j] = 1;
    	  }
      }

      return I;
   }

}
