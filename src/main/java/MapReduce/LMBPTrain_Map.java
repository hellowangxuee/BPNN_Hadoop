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
    private double MSE_upperbound=0.01;

    protected void getANNPath(Context context) throws IOException, InterruptedException {
        Configuration conf = context.getConfiguration();
        this.ANN_path = conf.get("ThisIterationPath");
    }

    public void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
        getANNPath(context);
        String line = value.toString();
        String[] LineDataArr = line.split(";");

        //establish a ANN from existing file
        ArtificialNeuralNetwork TrainingANN = new ArtificialNeuralNetwork(ANN_path);
        ArtificialNeuralNetwork BatchStore = new ArtificialNeuralNetwork(TrainingANN.getANN());
        BatchStore.clearNetwork();
        int ParaNum = 0;
        for (int i = 0; i < TrainingANN.getLayerNum(); i++) {
            ParaNum += TrainingANN.getANN()[i].getNeuronNum() * (TrainingANN.getANN()[i].getInputNum() + 1);
        }

        double ThisTimeSquErrSum = 0.0;
        double LastTimeSquErrSum = 0.0;
        double MSE = 0.0;
        double[][] ErrArr = new double[LineDataArr.length][1];
        Zmat TotalJacobianMatrix = null;
        Zmat[] JacobMatrixSeries = new Zmat[LineDataArr.length];
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
            JacobMatrixSeries[EntryNum] = TrainingANN.getJacobianMatrix();
        }
        MSE = ThisTimeSquErrSum / LineDataArr.length;
        context.write(new Text("SquareError"), new DoubleWritable(MSE));
        LastTimeSquErrSum = ThisTimeSquErrSum;
        TotalJacobianMatrix = new Zmat(LineDataArr.length, ParaNum);
        for (int k = 0; k < LineDataArr.length; k++) {
            for (int p = 0; p < ParaNum; p++) {
                TotalJacobianMatrix.put0(k, p, new Z(JacobMatrixSeries[k].get0(0, p)));
            }
        }
        try {
            for (int Ite = 0; MSE > MSE_upperbound && Ite < 20; Ite++) {
                Zmat H = Times.o(transpose.o(TotalJacobianMatrix), TotalJacobianMatrix);
                Zmat G = Plus.o(H, Times.o(new Z(miu, 0), Eye.o(H.nr)));
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
                    JacobMatrixSeries[EntryNum] = TrainingANN.getJacobianMatrix();
                }

                if (ThisTimeSquErrSum < LastTimeSquErrSum) {
                    miu /= MultipliedFactor;
                    //TotalJacobianMatrix = new Zmat(LineDataArr.length, ParaNum);
                    for (int k = 0; k < LineDataArr.length; k++) {
                        for (int p = 0; p < ParaNum; p++) {
                            TotalJacobianMatrix.put0(k, p, new Z(JacobMatrixSeries[k].get0(0, p)));
                        }
                    }
                    BatchStore.updateWeightNetwork(UpdatesANN.getANN());
                    LastTimeSquErrSum = ThisTimeSquErrSum;
                    MSE = ThisTimeSquErrSum / LineDataArr.length;
                } else {
                    miu *= MultipliedFactor;
                    TrainingANN.updateWeightNetwork(ArtificialNeuralNetwork.multiplyNeuronLayers(UpdatesANN.getANN(), -1));
                }
            }
        } catch (Exception e) {
            System.out.println(e.toString());
            context.write(new Text("Error"), new DoubleWritable(-1));
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

    }
}