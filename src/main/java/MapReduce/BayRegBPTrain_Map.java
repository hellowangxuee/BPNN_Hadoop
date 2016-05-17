package MapReduce;

import Jampack.*;
import NeuralNetwork.ANN_Train;
import NeuralNetwork.ArtificialNeuralNetwork;
import org.apache.commons.math3.analysis.function.Sin;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.xerces.dom.ElementNSImpl;

import java.io.IOException;
import java.util.Vector;

/**
 * Created by mlx on 4/25/16.
 */
public class BayRegBPTrain_Map extends
        Mapper<LongWritable, Text, Text, DoubleWritable> {
    private Text Windex = new Text();
    private Text Bindex = new Text();
    private String ANN_path = "";
    private double MSE_upperbound = 0.001;

    protected void getANNPath(Context context) throws IOException, InterruptedException {
        Configuration conf = context.getConfiguration();
        this.ANN_path = conf.get("ThisIterationPath");
    }

    public void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
        getANNPath(context);
        String line = value.toString();
        String[] LineDataArr = line.split(";");
//        for(int p=0;p<LineDataArr.length;p++){
//            System.out.println(LineDataArr[p]);
//        }
        double BETA = 1;
        double ALPHA = 0.01;

        ArtificialNeuralNetwork TrainingANN = new ArtificialNeuralNetwork(ANN_path);
        ArtificialNeuralNetwork TotalUpdates = new ArtificialNeuralNetwork(TrainingANN.getANN());
        TotalUpdates.clearNetwork();

        int ParaNum = 0;
        for (int i = 0; i < TrainingANN.getLayerNum(); i++) {
            ParaNum += TrainingANN.getANN()[i].getNeuronNum() * (TrainingANN.getANN()[i].getInputNum() + 1);
        }

        double GAMMA = ParaNum;
        double[][] ErrVec = new double[1][1];
        double[][] ForwardResult = null;
        double[][] ErrorArr = new double[LineDataArr.length][1];

        Zmat TotalJacobianMatrix = null;
        Zmat[] JacobMatrixSeries = new Zmat[LineDataArr.length];
        try {
            for (int Ite = 0; Ite < ParaNum; Ite++) {
                double SquareErrSum = 0.0;
                double WeightSquareSum = TrainingANN.getWeightSquareSum();
                for (int EntryNum = 0; EntryNum < LineDataArr.length; EntryNum++) {
                    String[] DataArr = LineDataArr[EntryNum].split("\t");
                    double Tag = Double.parseDouble(DataArr[DataArr.length - 1]);
                    double[][] InputVec = new double[DataArr.length - 1][1];
                    for (int i = 0; i < DataArr.length - 1; i++) {
                        InputVec[i][0] = Double.parseDouble(DataArr[i]);
                    }
                    ForwardResult = TrainingANN.getForwardResult(InputVec);
                    ErrVec[0][0] = Tag - ForwardResult[0][0];
                    SquareErrSum += Math.pow(ErrVec[0][0], 2);
                    ErrorArr[EntryNum][0] = ErrVec[0][0];
                    JacobMatrixSeries[EntryNum] = TrainingANN.getJacobianMatrix();
                }
                if (Ite == 0) {
                    context.write(new Text("SquareError"), new DoubleWritable(SquareErrSum / LineDataArr.length));
                }
                if (SquareErrSum / LineDataArr.length < MSE_upperbound) {
                    break;
                }
                TotalJacobianMatrix = new Zmat(LineDataArr.length, ParaNum);
                for (int k = 0; k < LineDataArr.length; k++) {
                    for (int p = 0; p < ParaNum; p++) {
                        TotalJacobianMatrix.put0(k, p, new Z(JacobMatrixSeries[k].get0(0, p)));
                    }
                }
                ALPHA = GAMMA / (2 * WeightSquareSum);
                BETA = (LineDataArr.length - GAMMA) / (2 * SquareErrSum);
                Zmat H = Times.o(transpose.o(TotalJacobianMatrix), TotalJacobianMatrix);
                Zmat G = Plus.o(Times.o(new Z(2 * BETA, 0), H), Times.o(new Z(2 * ALPHA, 0), Eye.o(H.nr)));
                GAMMA = ParaNum - 2 * ALPHA / Trace.o((G)).re;
//                    double miu = 0.01;
//                    Eig EigOfH = new Eig(H);
//                    double minEigValue = 2 * BETA * EigOfH.D.get0(H.nr - 1).re + 2 * ALPHA;
//                    if (minEigValue < 0 && miu + minEigValue < 0) {
//                        miu += (-minEigValue);
//                    }
//                    G = Plus.o(G, Times.o(new Z(miu, 0), Eye.o(H.nr)));
                Zmat NetworkUpdates = Times.o(new Z(-1, 0), Times.o(Times.o(Inv.o(G), transpose.o(TotalJacobianMatrix)), new Zmat(ErrorArr)));
                ArtificialNeuralNetwork UpdatesANN = new ArtificialNeuralNetwork(TrainingANN.getANN());
                int LayerIndex = 0;
                int NeuronIndex = 0;
                int InputIndex = 0;
                for (int index = 0; index < NetworkUpdates.nr; index++) {
                    int LayerParaSum = 0;
                    for (int i = 0; i < UpdatesANN.getLayerNum(); i++) {
                        LayerParaSum += UpdatesANN.getANN()[i].getNeuronNum() * (UpdatesANN.getANN()[i].getInputNum() + 1);
                        if (index / LayerParaSum < 1) {
                            LayerIndex = i;
                            break;
                        }
                    }
                    int NeuronOffset = index - LayerParaSum + UpdatesANN.getANN()[LayerIndex].getNeuronNum() * (UpdatesANN.getANN()[LayerIndex].getInputNum() + 1);
                    if (NeuronOffset >= (UpdatesANN.getANN()[LayerIndex].getNeuronNum() * UpdatesANN.getANN()[LayerIndex].getInputNum())) {
                        int BiasNum = NeuronOffset - (UpdatesANN.getANN()[LayerIndex].getNeuronNum() * UpdatesANN.getANN()[LayerIndex].getInputNum());
                        UpdatesANN.setCertainBias(LayerIndex, BiasNum, NetworkUpdates.get0(index, 0).re);
                    } else {
                        NeuronIndex = NeuronOffset / UpdatesANN.getANN()[LayerIndex].getInputNum();
                        InputIndex = NeuronOffset % UpdatesANN.getANN()[LayerIndex].getInputNum();
                        UpdatesANN.setCertainWeight(LayerIndex, NeuronIndex, InputIndex, NetworkUpdates.get0(index, 0).re);
                    }
                }
                TrainingANN.updateWeightNetwork(UpdatesANN.getANN());
                TotalUpdates.updateWeightNetwork(UpdatesANN.getANN());
            }
        } catch (Exception e) {
            System.out.println(e.toString());
            context.write(new Text("Error"), new DoubleWritable(-1));
        }
        for (int i = 0; i < TotalUpdates.getANN().length; i++) {
            for (int j = 0; j < TotalUpdates.getANN()[i].getNeuronNum(); j++) {
                for (int k = 0; k < TotalUpdates.getANN()[i].getInputNum(); k++) {
                    String WeightIndex = "W-" + String.valueOf(i) + "-";
                    WeightIndex += String.valueOf(j) + "-" + String.valueOf(k);
                    Windex.set(WeightIndex);
                    context.write(Windex, new DoubleWritable(TotalUpdates.getANN()[i].getCertainWeight(j, k)));
                }
            }
        }
        for (int i = 0; i < TotalUpdates.getANN().length; i++) {
            for (int j = 0; j < TotalUpdates.getANN()[i].getNeuronNum(); j++) {
                String BiasIndex = "B-" + String.valueOf(i) + "-";
                BiasIndex += String.valueOf(j);
                Bindex.set(BiasIndex);
                context.write(Bindex, new DoubleWritable(TotalUpdates.getANN()[i].getCertainBias(j)));
            }
        }

    }
}