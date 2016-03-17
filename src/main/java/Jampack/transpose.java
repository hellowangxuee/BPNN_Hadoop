package Jampack;

public class transpose {
    public static Zmat o(Zmat A)
    {

      Zmat B = new Zmat(A.ncol, A.nrow);
      for (int i=0; i<A.ncol; i++)
         for (int j=0; j<A.nrow; j++){
            B.re[i][j] = A.re[j][i];
            B.im[i][j] = A.im[j][i];
      }
      return B;
   }
}
