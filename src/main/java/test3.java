import FileIO.FileReadNWrite;
import Jampack.JampackException;
import MapReduce.BPTrain;
import NeuralNetwork.ArtificialNeuralNetwork;
import com.sun.javafx.binding.DoubleConstant;
import com.sun.nio.sctp.PeerAddressChangeNotification;
import org.apache.hadoop.fs.Path;

import java.io.IOException;
import java.util.Map;
import java.util.Random;

/**
 * Created by mlx on 4/2/16.
 */
public class test3 {
    public static void main(String[] args) throws JampackException, IOException {
//        for(int i=0;i<100;i++) {
//            System.out.println(getRandomNum(0, 1));
//        }

        String PathPrefix = "hdfs://Master:9000/TEP_Classify/TEP_moVLBP-v2/TEP_moVLBPClassify-";
        for (int i = 0; ; i++) {
            try {
                String path = PathPrefix + String.valueOf(i);
                Map<String, Double> map = BPTrain.getWeightUpdatesFromFile(path);
                double MSE=map.get("MeanSquareError");
                if (String.valueOf(MSE).equals("null")){
                    break;
                }
                System.out.println(String.valueOf(i) + "\t" + String.valueOf(map.get("MeanSquareError")));
                map.clear();
            }
            catch (Exception E){
                break;
            }
        }
        //HDFS_IO.ReadNWrite.hdfs_createDir("hdfs://Master:9000/FuncSimu/BayRegBP-v4/");
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
    public static int getRandomNum(int m,int n) {
        return (m + (int) (Math.random() * (n - m + 1)));
    }


}
