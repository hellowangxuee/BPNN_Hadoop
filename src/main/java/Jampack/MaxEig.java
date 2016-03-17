package Jampack;

/**
   Eig implements the eigenvalue-vector decomposition of
   of a square matrix.  Specifically given a diagonalizable
   matrix A, there is a matrix nonsingular matrix X such that
<pre>
*      D = X<sup>-1</sup> AX
</pre>
   is diagonal.  The columns of X are eigenvectors of A corresponding
   to the diagonal elements of D.  Eig implements X as a Zmat and
   D as a Zdiagmat.
<p>
   Warning: if A is defective rounding error will allow Eig to
   compute a set of eigevectors.  However, the matrix X will
   be ill conditioned.
@version Pre-alpha
@author G. W. Stewart
*/

public class MaxEig{

/** The matrix of eigevectors */
    public Zmat X;

/** The diagonal matrix of eigenvalues */
    public double D_max;

/**
   Creates an eigenvalue-vector decomposition of a square matrix A.

   @param       A The matrix whose decomposition is to be
                  computed
   @exception   JampackException
                  Thrown if A is not square. <br>
                  Passed from below.
*/                

   public MaxEig(Eig A,int x_col)
   throws JampackException{
	   double D_1 = A.D.re[0];
	   int flag = 0;
	   for (int i=0;i<x_col;i++){
		   if (A.D.re[i] > D_1){
			   D_1 = A.D.re[i];
			   flag = i;
		   }		   
	   }
	   D_max = D_1;
	   X = new Zmat(x_col,1);
	   X = A.X.get(1, x_col, (flag+1), (flag+1));
   }
}


