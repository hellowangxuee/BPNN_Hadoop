package FileIO;

import java.io.*;
import java.util.Vector;

/**
 * Created by mlx on 3/11/16.
 */
public class FileReadNWrite {
    public static Vector<Double[]> readTxtFile(String filePath) {
        Vector<Double[]> vet = new Vector();
        try {
            String encoding = "GBK";
            File file = new File(filePath);
            if (file.isFile() && file.exists()) { //判断文件是否存在
                InputStreamReader read = new InputStreamReader(
                        new FileInputStream(file), encoding);//考虑到编码格式
                BufferedReader bufferedReader = new BufferedReader(read);
                String lineTxt = null;
                while ((lineTxt = bufferedReader.readLine()) != null) {
                    String[] lineArr = lineTxt.split("\t");
                    Double[] InputPair = new Double[lineArr.length];
                    for (int k = 0; k < lineArr.length; k++) {
                        InputPair[k] = Double.parseDouble(lineArr[k]);
                    }
                    vet.add(InputPair);
                }
                read.close();
            } else {
                System.out.println("找不到指定的文件");
            }
        } catch (Exception e) {
            System.out.println("读取文件内容出错");
            e.printStackTrace();
        }
        return vet;
    }
    public static void LocalWriteFile(String filePath,String[] content) throws IOException{
        try {
            //打开一个写文件器，构造函数中的第二个参数true表示以追加形式写文件
            FileWriter writer = new FileWriter(filePath, true);
            for(int i=0;i<content.length;i++) {
                if (content[i]!=null) {
                    writer.write(content[i] + "\n");
                }
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
    public static Vector LocalReadFile(String filePath){
        Vector vet = new Vector();
        try {
            String encoding = "GBK";
            File file = new File(filePath);
            if (file.isFile() && file.exists()) { //判断文件是否存在
                InputStreamReader read = new InputStreamReader(
                        new FileInputStream(file), encoding);//考虑到编码格式
                BufferedReader bufferedReader = new BufferedReader(read);
                String lineTxt = null;
                while ((lineTxt = bufferedReader.readLine()) != null) {
                    vet.add(lineTxt);
                }
                read.close();
            } else {
                System.out.println("找不到指定的文件");
            }
        } catch (Exception e) {
            System.out.println("读取文件内容出错");
            e.printStackTrace();
        }
        return vet;

    }
}
