package NeuralNetwork;

/**
 * Created by Jackie on 16/3/3.
 */
public class TransFunc {

    public static double Log_Sigmoid(double n){
        double result=1.0/(1+Math.exp(0-n));
        return result;
    }

    public static int Symmetrical_Hard_Limit(double n){
        if(n>=0){
            return 1;
        }
        else{
            return -1;
        }
    }

    public static double Pure_Linear(double n){
        return n;
    }

    public static int Pure_Linear(int n){
        return n;
    }

    public static double ReLu(double n){
        return Math.log(1+Math.exp(n));
    }
}
