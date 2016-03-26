import FileIO.FileReadNWrite;
import HDFS_IO.ReadNWrite;
import Jampack.JampackException;
import NeuralNetwork.ArtificialNeuralNetwork;

import java.io.IOException;
import java.util.Vector;

/**
 * Created by mlx on 3/18/16.
 */
public class test2 {
    public static void main(String[] args) throws JampackException,IOException {

//        for(double Threshold=0.001;Threshold<0.5;Threshold+=0.001) {
////            int NormalRightNum=0;
////            int NormalWrongNum=0;
////            int Fault1RightNum=0;
////            int Fault1WrongNum=0;
//            int RightNum=0;
//            int WrongNum=0;
//            for (int i = 0; i < InputPair.size(); i++) {
//                temp = (double[]) (InputPair.get(i));
//                for (int k = 0; k < temp.length - 1; k++) {
//                    InputVec[k][0] = temp[k];
//                }
//                ForwardResult = fANN.getForwardResult(InputVec);
//                int TrueFlag=(int)(temp[temp.length-1]);
//                int PreFlag=0;
//                if(ForwardResult[0][0]>=Threshold){
//                    PreFlag=1;
//                }
//                if(TrueFlag==PreFlag){
//                    RightNum+=1;
//                }
//                else{
//                    WrongNum+=1;
//                }
////                System.out.println(String.valueOf(i) + "\t" + String.valueOf(temp[temp.length - 1]) + "\t" + String.valueOf(ForwardResult[0][0]));
//            }
//            double Accuracy=(double)(RightNum)/(RightNum+WrongNum);
//            System.out.println(String.valueOf(Threshold)+"\t"+String.valueOf(Accuracy));
//        }

//        double Accuracy=getAccuracy(fANN,"/home/mlx/Documents/TestingDataset",0.05);
//        System.out.println(String.valueOf(0.05)+"\t"+String.valueOf(Accuracy));
//        String ANNpath="hdfs://Master:9000/user/mlx/FinalANN-v1.3";
//        ArtificialNeuralNetwork fANN=new ArtificialNeuralNetwork(ANNpath);
//        String[] ann=fANN.saveANN();
//        ReadNWrite.hdfs_Write(ann,"hdfs://Master:9000/user/mlx/FinalANN-v1.4");
        String a="WB-ST-T";
        if(a.contains("W")){
            System.out.println("1");
        }
    }

    public static double getAccuracy(ArtificialNeuralNetwork testANN,String datasetPath,double Threshold) {
        Vector dataArr = FileReadNWrite.readTxtFile(datasetPath);
        double[] temp = (double[]) dataArr.get(0);
        double[][] InputVec = new double[temp.length - 1][1];
        int RightNum = 0;
        int WrongNum = 0;
        for (int i = 0; i < dataArr.size(); i++) {
            temp = (double[]) (dataArr.get(i));
            for (int k = 0; k < temp.length - 1; k++) {
                InputVec[k][0] = temp[k];
            }
            double[][] ForwardResult = testANN.getForwardResult(InputVec);
            int TrueFlag = (int) (temp[temp.length - 1]);
            int PreFlag = 0;
            if (ForwardResult[0][0] >= Threshold) {
                PreFlag = 1;
            }
            if (TrueFlag == PreFlag) {
                RightNum += 1;
            } else {
                WrongNum += 1;
            }
        }
        double Accuracy = (double) (RightNum) / (RightNum + WrongNum);
        return Accuracy;
    }
}
