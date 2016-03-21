import HDFS_IO.ReadNWrite;
import Jampack.JampackException;

import java.io.IOException;

/**
 * Created by mlx on 3/18/16.
 */
public class test2 {
    public static void main(String[] args) throws JampackException,IOException {
        String ipPrefix="hdfs://localhost:9000/user/BP_EXP3/TrainGeneration-";
        for(int i=0;i<150;i++){
            String path=ipPrefix+String.valueOf(i)+"/part-r-00000";
            String[] ContentArr=ReadNWrite.hdfs_Read(path);
            System.out.println(String.valueOf(i)+"\t"+ContentArr[3].split("\t")[1]);
        }

    }
}
