import Jampack.JampackException;
import MapReduce.BPTrain;
import NeuralNetwork.ArtificialNeuralNetwork;
import com.sun.nio.sctp.PeerAddressChangeNotification;

import java.io.IOException;
import java.util.Map;
import java.util.Random;

/**
 * Created by mlx on 4/2/16.
 */
public class test3 {
    public static void main(String[] args) throws Exception{
        for(int i=0;i<1000;i++){
            Random random=new Random();
            System.out.println(random.nextGaussian());
        }
//        String PathPrefix = "hdfs://Master:9000/user/mlx/temex_TEST1/SDBP_MOVL-";
//        for (int i = 0; i < 13; i++) {
//            String path = PathPrefix + String.valueOf(i);
//            Map<String, Double> map = BPTrain.getWeightUpdatesFromFile(path);
//            System.out.println(String.valueOf(i) + "\t" + String.valueOf(map.get("MeanSquareError")));
//        }
//        int InputNum = 41;
//        int LayerNum = 3;
//        int[] NumEachLayer = {7, 10, 1};
//        int[] IndexEachLayer = {1, 4, 4};
//        ArtificialNeuralNetwork FinalANN = new ArtificialNeuralNetwork(InputNum, LayerNum, NumEachLayer, IndexEachLayer);
//        String[] F= FinalANN.saveANN();
//        for(int i=0;i<F.length;i++){
//            System.out.println(F[i]);
//        }
    }
}
