package MapReduce;

import Jampack.*;
import Jampack.Times;
import NeuralNetwork.ArtificialNeuralNetwork;
import org.apache.commons.digester.plugins.PluginAssertionFailure;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.util.Time;
import org.apache.hadoop.yarn.util.*;

import java.io.IOException;
import java.util.StringTokenizer;
import java.util.Vector;
import java.util.jar.JarException;

/**
 * Created by mlx on 4/4/16.
 */
public class LMBPTrain_Map extends
        Mapper<LongWritable, Text, Text, DoubleWritable> {

    private Text Windex = new Text();
    private Text Bindex = new Text();
    private String ANN_path = "";
    private double miu = 0.01;
    private double MultipliedFactor = 10;
    private double MSE_upperbound=0.001;

    protected void getANNPath(Context context) throws IOException, InterruptedException {
        Configuration conf = context.getConfiguration();
        this.ANN_path = conf.get("ThisIterationPath");
    }

    public void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
        getANNPath(context);

        //pure text to String
        String line = value.toString();
        // 将输入的数据首先按行进行分割
        String[] LineDataArr = line.split("\n");

        //establish a ANN from existing file
        ArtificialNeuralNetwork TrainingANN = new ArtificialNeuralNetwork(ANN_path);
        ArtificialNeuralNetwork BatchStore = new ArtificialNeuralNetwork(TrainingANN.getANN());
        BatchStore.clearNetwork();

        double ThisTimeSquErrSum = 0.0;
        double LastTimeSquErrSum = 0.0;
        double MSE = 0.0;
        //Vector<Double[]> InputDataVector=new Vector<Double[]>();
        double[][] ErrArr = new double[LineDataArr.length][1];
        Zmat TotalJacobianMatrix = null;
        double[][] ErrVec = new double[TrainingANN.getOutputNum()][1];

        for (int EntryNum = 0; EntryNum < LineDataArr.length; EntryNum++) {
            String[] DataArr = LineDataArr[EntryNum].split("\t");
            double Tag = Double.parseDouble(DataArr[DataArr.length - 1]);
            double[][] InputVec = new double[DataArr.length - 1][1];
            for (int i = 0; i < DataArr.length - 1; i++) {
                InputVec[i][0] = Double.parseDouble(DataArr[i]);
            }
            double[][] ForwardResult = TrainingANN.getForwardResult(InputVec);
            ErrVec[0][0] = Tag - ForwardResult[0][0];
            ThisTimeSquErrSum += Math.pow(ErrVec[0][0], 2);

            ErrArr[EntryNum][0] = ErrVec[0][0];
            context.write(new Text("SquareError"), new DoubleWritable(ErrVec[0][0] * ErrVec[0][0]));
            if (EntryNum == 0) {
                TotalJacobianMatrix = TrainingANN.getJacobianMatrix();
            } else {
                try {
                    TotalJacobianMatrix = Merge.o21(TotalJacobianMatrix, TrainingANN.getJacobianMatrix());
                } catch (Exception e) {
                    System.out.println(e.toString());
                    continue;
                }
            }
        }
        MSE = ThisTimeSquErrSum / LineDataArr.length;
        LastTimeSquErrSum = ThisTimeSquErrSum;
        for (; MSE > MSE_upperbound; ) {
            try {
                Zmat H = Times.o(transpose.o(TotalJacobianMatrix), TotalJacobianMatrix);
                int OrderOfH = H.nr;
                Eig EigOfH = new Eig(H);
                double minEigValue = EigOfH.D.get0(OrderOfH-1).re;
                if (minEigValue < 0 && miu + minEigValue < 0) {
                    miu += (-minEigValue);
                }
                Zmat G = Plus.o(H, Times.o(new Z(miu, 0), Eye.o(OrderOfH)));
                Zmat NetworkUpdates = Times.o(new Z(-1, 0), Times.o(Times.o(Inv.o(G), transpose.o(TotalJacobianMatrix)), new Zmat(ErrArr)));
                ArtificialNeuralNetwork UpdatesANN = new ArtificialNeuralNetwork(TrainingANN.getANN());

                int LayerNum = 0;
                int NeuronNum = 0;
                int InputNum = 0;
                for (int index = 0; index < NetworkUpdates.nr; index++) {
                    int LayerParaSum = 0;
                    for (int i = 0; i < UpdatesANN.getLayerNum(); i++) {
                        LayerParaSum += UpdatesANN.getANN()[i].getNeuronNum() * (UpdatesANN.getANN()[i].getInputNum() + 1);
                        if (index / LayerParaSum < 1) {
                            LayerNum = i;
                            break;
                        }
                    }
                    int NeuronOffset = index - LayerParaSum + UpdatesANN.getANN()[LayerNum].getNeuronNum() * (UpdatesANN.getANN()[LayerNum].getInputNum() + 1);
                    if (NeuronOffset >= (UpdatesANN.getANN()[LayerNum].getNeuronNum() * UpdatesANN.getANN()[LayerNum].getInputNum())) {
                        int BiasNum = NeuronOffset - (UpdatesANN.getANN()[LayerNum].getNeuronNum() * UpdatesANN.getANN()[LayerNum].getInputNum());
                        UpdatesANN.setCertainBias(LayerNum, BiasNum, NetworkUpdates.get0(index, 0).re);
                    } else {
                        NeuronNum = NeuronOffset / UpdatesANN.getANN()[LayerNum].getInputNum();
                        InputNum = NeuronOffset % UpdatesANN.getANN()[LayerNum].getInputNum();
                        UpdatesANN.setCertainWeight(LayerNum, NeuronNum, InputNum, NetworkUpdates.get0(index, 0).re);
                    }
                }
                TrainingANN.updateWeightNetwork(UpdatesANN.getANN());
                Zmat TempJacobian = null;
                ThisTimeSquErrSum = 0;
                for (int EntryNum = 0; EntryNum < LineDataArr.length; EntryNum++) {
                    String[] DataArr = LineDataArr[EntryNum].split("\t");
                    double Tag = Double.parseDouble(DataArr[DataArr.length - 1]);
                    double[][] InputVec = new double[DataArr.length - 1][1];
                    for (int i = 0; i < DataArr.length - 1; i++) {
                        InputVec[i][0] = Double.parseDouble(DataArr[i]);
                    }
                    double[][] ForwardResult = TrainingANN.getForwardResult(InputVec);

                    ErrVec[0][0] = Tag - ForwardResult[0][0];
                    ThisTimeSquErrSum += Math.pow(ErrVec[0][0], 2);
                    ErrArr[EntryNum][0] = ErrVec[0][0];
                    if (EntryNum == 0) {
                        TempJacobian = TrainingANN.getJacobianMatrix();
                    } else {
                        TempJacobian = Merge.o21(TempJacobian, TrainingANN.getJacobianMatrix());
                    }
                }
                if (ThisTimeSquErrSum < LastTimeSquErrSum) {
                    miu /= MultipliedFactor;
                    TotalJacobianMatrix = new Zmat(TempJacobian);
                    BatchStore.updateWeightNetwork(UpdatesANN.getANN());
                    LastTimeSquErrSum = ThisTimeSquErrSum;
                    MSE = ThisTimeSquErrSum / LineDataArr.length;
                } else {
                    miu *= MultipliedFactor;
                    TrainingANN.updateWeightNetwork(ArtificialNeuralNetwork.multiplyNeuronLayers(UpdatesANN.getANN(), -1));
                }

            } catch (Exception e) {
                System.out.println(e.toString());
                break;
            }
        }
        for (int i = 0; i < BatchStore.getANN().length; i++) {
            for (int j = 0; j < BatchStore.getANN()[i].getNeuronNum(); j++) {
                for (int k = 0; k < BatchStore.getANN()[i].getInputNum(); k++) {
                    String WeightIndex = "W-" + String.valueOf(i) + "-";
                    WeightIndex += String.valueOf(j) + "-" + String.valueOf(k);
                    Windex.set(WeightIndex);
                    context.write(Windex, new DoubleWritable(BatchStore.getANN()[i].getCertainWeight(j, k)));
                }
            }
        }
        for (int i = 0; i < BatchStore.getANN().length; i++) {
            for (int j = 0; j < BatchStore.getANN()[i].getNeuronNum(); j++) {
                String BiasIndex = "B-" + String.valueOf(i) + "-";
                BiasIndex += String.valueOf(j);
                Bindex.set(BiasIndex);
                context.write(Bindex, new DoubleWritable(BatchStore.getANN()[i].getCertainBias(j)));
            }
        }
        //BatchStore.clearNetwork();
//        while(MSE > 0.001) {
//            Zmat TotalJacobianMatrix = null;
//            double[][] ErrArr=new double[LineDataArr.length][1];
//            for (int EntryNum = 0; EntryNum < LineDataArr.length; EntryNum++) {
//                String[] DataArr = LineDataArr[EntryNum].split("\t");
//                //Double[] WholeInput = new Double[DataArr.length];
//
//                double Tag = Double.parseDouble(DataArr[DataArr.length - 1]);
//                double[][] InputVec = new double[DataArr.length - 1][1];
//                for (int i = 0; i < DataArr.length - 1; i++) {
//                    InputVec[i][0] = Double.parseDouble(DataArr[i]);
//                    //WholeInput[i] = InputVec[i][0];
//                }
//                //WholeInput[DataArr.length - 1] = Tag;
//                //InputDataVector.add(WholeInput);
//
//                double[][] ForwardResult = TrainingANN.getForwardResult(InputVec);
//                double[][] ErrVec = new double[TrainingANN.getOutputNum()][1];
//                ErrVec[0][0] = Tag - ForwardResult[0][0];
//                ThisTimeSquErrSum += Math.pow(ErrVec[0][0], 2);
//                ErrArr[EntryNum][0]=ErrVec[0][0];
//
//                if (EntryNum == 0) {
//                    TotalJacobianMatrix = TrainingANN.getJacobianMatrix();
//                } else {
//                    try {
//                        TotalJacobianMatrix = Merge.o21(TotalJacobianMatrix, TrainingANN.getJacobianMatrix());
//                    } catch (Exception e) {
//                        System.out.println(e.toString());
//                        continue;
//                    }
//                }
//            }
//            MSE=ThisTimeSquErrSum/LineDataArr.length;
//            for(int TurnTime=0;true;TurnTime++) {
//                if (TurnTime!=0 && ThisTimeSquErrSum < LastTimeSquErrSum) {
//                    this.miu /= this.MultipliedFactor;
//                    break;
//                } else {
//                    if(TurnTime!=0) {
//                        this.miu *= this.MultipliedFactor;
//                    }
//                    try {
//                        Zmat H = Times.o(transpose.o(TotalJacobianMatrix), TotalJacobianMatrix);
//                        int OrderOfH = H.nr;
//                        Eig EigOfH = new Eig(H);
//                        double minEigValue = Double.MAX_VALUE;
//                        for (int k = 0; k < OrderOfH; k++) {
//                            if (EigOfH.D.get0(k).re < minEigValue) {
//                                minEigValue = EigOfH.D.get0(k).re;
//                            }
//                        }
//                        if (minEigValue < 0) {
//                            miu += (-minEigValue);
//                        }
//                        Zmat G = Plus.o(H, Times.o(new Z(miu, 0), Eye.o(OrderOfH)));
//                        Zmat NetworkUpdates = Times.o(new Z(-1, 0), Times.o(Times.o(Inv.o(G), transpose.o(TotalJacobianMatrix)), new Zmat(ErrArr)));
//                        ArtificialNeuralNetwork UpdatesANN = new ArtificialNeuralNetwork(TrainingANN.getANN());
//
//                        int LayerNum = 0;
//                        int NeuronNum = 0;
//                        int InputNum = 0;
//                        for (int index = 0; index < NetworkUpdates.nr; index++) {
//                            int LayerParaSum = 0;
//                            for (int i = 0; i < UpdatesANN.getLayerNum(); i++) {
//                                LayerParaSum += UpdatesANN.getANN()[i].getNeuronNum() * (UpdatesANN.getANN()[i].getInputNum() + 1);
//                                if (index / LayerParaSum < 1) {
//                                    LayerNum = i;
//                                    break;
//                                }
//                            }
//                            int NeuronOffset = index - LayerParaSum + UpdatesANN.getANN()[LayerNum].getNeuronNum() * (UpdatesANN.getANN()[LayerNum].getInputNum() + 1);
//                            if (NeuronOffset >= (UpdatesANN.getANN()[LayerNum].getNeuronNum() * UpdatesANN.getANN()[LayerNum].getInputNum())) {
//                                int BiasNum = NeuronOffset - (UpdatesANN.getANN()[LayerNum].getNeuronNum() * UpdatesANN.getANN()[LayerNum].getInputNum());
//                                UpdatesANN.setCertainBias(LayerNum, BiasNum, NetworkUpdates.get0(index, 0).re);
//                            } else {
//                                NeuronNum = NeuronOffset / UpdatesANN.getANN()[LayerNum].getInputNum();
//                                InputNum = NeuronOffset % UpdatesANN.getANN()[LayerNum].getInputNum();
//                                UpdatesANN.setCertainWeight(LayerNum, NeuronNum, InputNum, NetworkUpdates.get0(index, 0).re);
//                            }
//                        }
//                        TrainingANN.updateWeightNetwork(UpdatesANN.getANN());
//                        BatchStore.updateWeightNetwork(UpdatesANN.getANN());
//
//                    } catch (Exception e) {
//                        System.out.println(e.toString());
//                        continue;
//                    }
//                    LastTimeSquErrSum = ThisTimeSquErrSum;
//                }
//            }
//        }
    }
}
