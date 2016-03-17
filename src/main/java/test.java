import Jampack.JampackException;
import Jampack.Zmat;

import NeuralNetwork.ArtificialNeuralNetwork;
import NeuralNetwork.NeuronLayer;
import NeuralNetwork.SingleNeuron;
import NeuralNetwork.TransFunc;

import java.net.URI;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.omg.Messaging.SYNC_WITH_TRANSPORT;

import java.io.*;
import java.nio.DoubleBuffer;
import java.util.*;

import HDFS_IO.*;
/**
 * Created by Jackie on 16/3/3.
 */
public class test {
    public static void main(String[] args) throws JampackException,Exception {
//        double[][] A = {{1, 2, 3}, {2, 3, 4}, {4, 5, 6}};
//        double[][] B = {{3, 2, 2.5}, {1.3, 3.2, 2.3}, {5.2, 1.2, 4}};
//        Zmat matA = new Zmat(A);
//        Zmat matB = new Zmat(B);
//        Zmat matC = Jampack.mult.o(matA, matB);
//        for(int i=0;i<100;i++) {
//            double a = getRan();
//            System.out.println(a);
//        }
//        Vector InputPair=readTxtFile("/Users/Jackie/Downloads/hadoop-2.6.0/INPUT/FuncInput.txt");
//
//        HashSet h=new HashSet();
//
//
//        int InputNum=1;
//        int LayerNum=2;
//        int[] NumEachLayer={3,1};
//        int[] IndexEachLayer={1,3};
//
//        ArtificialNeuralNetwork TestWork=new ArtificialNeuralNetwork(InputNum,LayerNum,NumEachLayer,IndexEachLayer);
//
//        ArtificialNeuralNetwork BatchStoreWork=new ArtificialNeuralNetwork(InputNum,LayerNum,NumEachLayer,IndexEachLayer);
//        BatchStoreWork.clearNetwork();
//
//        double[][] InputVec=new double[InputNum][1];
//        double[][] ErrVec=new double[NumEachLayer[NumEachLayer.length-1]][1];
//        double[][] ForwardResult=new double[NumEachLayer[NumEachLayer.length-1]][1];
//        double[] temp;
//        double ErrSum=0.0;
//
//
//        for(int t=0;t<20;t++) {
//            int RandomNum=getRandomNum(0, InputPair.size() - 1);
//            h.add(RandomNum);
//            for (int i = 0; i < InputPair.size(); i++) {
//                while (!h.contains(RandomNum)) {
//                    RandomNum = getRandomNum(0, InputPair.size() - 1);
//                    h.add(RandomNum);
//                }
//                temp = (double[]) (InputPair.get(RandomNum));
//                InputVec[0][0] = temp[0];
//                ForwardResult = TestWork.getForwardResult(InputVec);
//                ErrVec[0][0] = temp[1] - ForwardResult[0][0];
//                ErrSum += ErrVec[0][0];
//
//                TestWork.updateWeightNetwork(TestWork.getBackwardChange(ErrVec, 0.1));
//                RandomNum = getRandomNum(0, InputPair.size() - 1);
//                h.add(RandomNum);
//            }
//            h.clear();
//        }
//
//        Vector TestPair=readTxtFile("/Users/Jackie/Downloads/hadoop-2.6.0/INPUT/FuncTest.txt");
//        double MSE=0.0;
//        for(int i=0;i<TestPair.size();i++) {
//            temp = (double[]) (TestPair.get(i));
//            InputVec[0][0] = temp[0];
//            ForwardResult = TestWork.getForwardResult(InputVec);
//            MSE+=(temp[1] - ForwardResult[0][0])*(temp[1] - ForwardResult[0][0]);
//            System.out.println(ForwardResult[0][0]);
//        }
//
//        System.out.println("\n");
//        System.out.println(MSE);
//
//        ArtificialNeuralNetwork TestAnn=new ArtificialNeuralNetwork("/home/mlx/Documents/testANN");
//        int a=1;
        String uri = "hdfs://localhost:9000/user/mlx/testANN";
        String des = "hdfs://localhost:9000/user/mlx/testANN-v1";
        //读取hadoop文件系统的配置
//        Configuration conf = new Configuration();
//        conf.set("hadoop.job.ugi", "hadoop-user,hadoop-user");
//        //FileSystem是用户操作HDFS的核心类，它获得URI对应的HDFS文件系统
//        FileSystem fs = FileSystem.get(URI.create(uri),conf);
//        FSDataInputStream in = null;
//        try{
//            in = fs.open( new Path(uri) );
//            byte[] ContentBuff=new byte[4096];
//            in.read(ContentBuff);
//            String teststr = "";
//            for (byte b: ContentBuff) {
//                if (b != 0) {
//                    teststr += (char)b;
//                }
//            }
//            String[] testArr=teststr.split("\n");
//
//            System.out.println(teststr);
//
//        }
//        finally{
//            IOUtils.closeStream(in);
//        }
        String[] Content = ReadNWrite.hdfs_Read(uri);
        ReadNWrite.hdfs_Write(Content, des);

    }
    public static Vector readTxtFile(String filePath){
        Vector vet =new Vector();
        try {

            String encoding="GBK";
            File file=new File(filePath);
            if(file.isFile() && file.exists()){ //判断文件是否存在
                InputStreamReader read = new InputStreamReader(
                        new FileInputStream(file),encoding);//考虑到编码格式
                BufferedReader bufferedReader = new BufferedReader(read);
                String lineTxt = null;
                while((lineTxt = bufferedReader.readLine()) != null){
                    double[] InputPair=new double[2];
                    String[] lineArr=lineTxt.split("\t");
                    InputPair[0]=Double.parseDouble(lineArr[0]);
                    InputPair[1]=Double.parseDouble(lineArr[1]);
                    vet.add(InputPair);
                }
                read.close();
            }else{
                System.out.println("找不到指定的文件");
            }
        } catch (Exception e) {
            System.out.println("读取文件内容出错");
            e.printStackTrace();
        }
        return vet;
    }

    public static int getRandomNum(int m,int n) {
        return (m + (int) (Math.random() * (n - m + 1)));
    }

}
