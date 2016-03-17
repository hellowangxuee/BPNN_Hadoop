package Jampack;

/**
   Calculate the product of two matrices.

   @version Pre-alpha
   @author K. W. Steven
*/

public class mult{

/**
   Computes the product of two Zmats
   @param     A  The first Zmat
   @param     B  The second Zmat
   @return    A * B
   @exception JampackException
              Thrown for nonconformity.
*/
   public static Zmat o(Zmat A, Zmat B)
   throws JampackException{
      if (A.ncol!=B.nrow ){
         throw new JampackException("Matrices not conformable for multiplication");
      }
      Zmat C = new Zmat(A.nr, B.nc);

      for (int i=0; i<A.nrow; i++)
         for (int j=0; j<B.ncol; j++)
         	for(int k=0; k<A.ncol;k++){
         		C.re[i][j] = C.re[i][j] + A.re[i][k] * B.re[k][j];
         		C.im[i][j] = C.im[i][j] + A.im[i][k] * B.im[k][j];
         }
      return C;
   }

/**
   Computes the sum of a Zmat and a Zdiagmat.
   @param     A The Zmat
   @param     D The Zdiagmat
   @return    A + D
   @exception JampackException
              Thrown for nonconformity.
*/

   public static Zmat o(Zmat A, Zdiagmat D)
   throws JampackException{

      if (D.order != A.nrow || D.order != A.ncol){
         throw new JampackException("Matrices not conformable for addition");
      }
      Zmat C = new Zmat(A);
      for (int i=0; i<A.nrow; i++){
         C.re[i][i] = C.re[i][i] + D.re[i];
         C.im[i][i] = C.im[i][i] + D.im[i];
      }
      return C;
   }

/**
   Computes the sum of a Zdiagmat and a Zmat.
   @param     D  The Zdiagmat
   @param     A  The Zmat
   @return    D + A
   @exception JampackException
              Thrown for nonconformity.
*/

   public static Zmat o(Zdiagmat D, Zmat A)
   throws JampackException{

      if (D.order != A.nrow || D.order != A.ncol){
         throw new JampackException("Matrices not conformable for addition");
      }
      Zmat C = new Zmat(A);
      for (int i=0; i<D.order; i++){
         C.re[i][i] = C.re[i][i] + D.re[i];
         C.im[i][i] = C.im[i][i] + D.im[i];
      }
      return C;
   }

/**
   Computes the sum of a Zdiagmat and a Zdiagmat.
   @param     D1  The first Zdiagmat
   @param     D2  The second Zdiagmat
   @return    D1 + D2
   @exception JampackException
              Thrown for nonconformity.
*/

   public static Zdiagmat o(Zdiagmat D1, Zdiagmat D2)
   throws JampackException{

      if (D1.order != D2.order){
         throw new JampackException("Matrices not conformable for addition");
      }
      Zdiagmat C = new Zdiagmat(D1);
      for (int i=0; i<D1.order; i++){
         C.re[i] = C.re[i] + D2.re[i];
         C.im[i] = C.im[i] + D2.im[i];
      }
      return C;
   }
}
