import Jampack.JampackException;
import MapReduce.BPTrain;

import java.io.IOException;
import java.util.Map;

/**
 * Created by mlx on 4/2/16.
 */
public class test3 {
    public static void main(String[] args) throws Exception {
        String PathPrefix = "hdfs://Master:9000/user/mlx/Simu_EXP8/CGBP_G-";
        for (int i = 0; i < 50; i++) {
            String path = PathPrefix + String.valueOf(i);
            Map<String, Double> map = BPTrain.getWeightUpdatesFromFile(path);
            System.out.println(String.valueOf(i) + "\t" + String.valueOf(map.get("MeanSquareError")));
        }

    }
}
