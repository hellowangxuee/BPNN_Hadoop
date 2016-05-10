package TEP_Classification;

import Jampack.JampackException;

import java.io.IOException;
import java.util.Vector;

import static FileIO.FileReadNWrite.readTxtFile;

/**
 * Created by mlx on 5/9/16.
 */
public class PredictAccuracyCal {
    public static void main(String[] args) throws JampackException, IOException {
        Vector<Double[]> TestData = readTxtFile("/home/mlx/Documents/DataSet/TEP_TestData");
        Vector<Double[]> ResultData = readTxtFile("/home/mlx/Documents/ProcessResult/TEP_CGBP_PreRe");

        for (double ClassThreshold = -0.0001; ClassThreshold < 1.1; ClassThreshold += 0.01) {
            int PreRightNum = 0;
            int PreWrongNum = 0;
            for (int i = 0; i < TestData.size(); i++) {
                Double[] OriginOneEntry = (TestData.get(i));
                Double[] PreOneEntry = ResultData.get(i);
                double Tag = OriginOneEntry[OriginOneEntry.length - 1];
                double PreTag = PreOneEntry[0];
                if ((PreTag >= ClassThreshold && Tag == 1.0) || (PreTag < ClassThreshold && Tag == 0.0)) {
                    PreRightNum++;
                } else {
                    PreWrongNum++;
                }
            }
            double Accuracy = ((double) PreRightNum) / (TestData.size());
            System.out.println(ClassThreshold + "\t" + PreRightNum + "\t" + PreWrongNum + "\t" + Accuracy);
        }
    }
}
