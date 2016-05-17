package HDFS_IO;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.hdfs.protocol.datatransfer.PacketHeader;
import org.apache.hadoop.io.IOUtils;
import org.apache.log4j.Logger;
import org.apache.hadoop.fs.Path;
import sun.security.krb5.internal.crypto.Des;

import java.io.*;
import java.util.Vector;

import java.net.URI;

/**
 * Created by mlx on 3/17/16.
 */
public class ReadNWrite {
    static Logger logger = null;


    public static boolean hdfs_isFileExist(String filePath) throws IOException {
        Configuration conf = new Configuration();

        conf.addResource(new Path("/usr/local/hadoop/etc/hadoop/core-site.xml"));
        conf.addResource(new Path("/usr/local/hadoop/etc/hadoop/hdfs-site.xml"));

        FileSystem fs = FileSystem.get(conf);
        Path inFile = new Path(filePath);
        if (!fs.exists(inFile)) {
            return false;
        } else {
            return true;
        }
    }
    public static boolean hdfs_createDir(String filePath) throws IOException {
        Configuration conf = new Configuration();

        conf.addResource(new Path("/usr/local/hadoop/etc/hadoop/core-site.xml"));
        conf.addResource(new Path("/usr/local/hadoop/etc/hadoop/hdfs-site.xml"));

        FileSystem fs = FileSystem.get(conf);
        Path f=new Path(filePath);
        if(!fs.exists(f)){
            fs.mkdirs(f);
            return true;
        }
        else{
            return false;
        }
    }

    public static boolean hdfs_delete(String filePath) throws IOException {
        Configuration conf = new Configuration();

        conf.addResource(new Path("/usr/local/hadoop/etc/hadoop/core-site.xml"));
        conf.addResource(new Path("/usr/local/hadoop/etc/hadoop/hdfs-site.xml"));

        FileSystem fs = FileSystem.get(conf);
        Path f=new Path(filePath);
        if(fs.exists(f)){
            fs.delete(f,true);
            return true;
        }
        else{
            return false;
        }
    }



    public static Vector hdfs_Read(String filePath) throws IOException {
        Vector vet = new Vector();
        Configuration conf = new Configuration();
        conf.set("hadoop.job.ugi", "hadoop-user,hadoop-user");
        FileSystem fs = FileSystem.get(URI.create(filePath), conf);
        FSDataInputStream in = fs.open(new Path(filePath));
        BufferedReader bis = new BufferedReader(new InputStreamReader(in, "GBK"));
        String temp;
        try {
            while ((temp = bis.readLine()) != null) {
                vet.add(temp);
            }
        } catch (Exception e) {
            logger.error("ReadHDFS_Error:\t" + e.toString());
        } finally {
//            bis.close();
//            fs.close();
            IOUtils.closeStream(in);
            return vet;
        }
    }

    public static boolean hdfs_Write(String[] ContentArr, String DesPath) throws IOException {
        logger = Logger.getLogger(ReadNWrite.class.getName());

        boolean SuccOrNot = true;

        Configuration conf = new Configuration();
        conf.set("hadoop.job.ugi", "hadoop-user,hadoop-user");

        FileSystem fs = FileSystem.get(URI.create(DesPath), conf);
        Path outFile = new Path(DesPath);

        if (fs.exists(outFile)) {
            logger.error("WriteHDFS_Error:\tOutput already exists.");
            return false;
        }
        String TotalContent = "";
        for (int i = 0; i < ContentArr.length; i++) {
            TotalContent += ContentArr[i] + "\n";
        }
        byte TotalContenByte[] = TotalContent.getBytes();
        FSDataOutputStream out = fs.create(outFile);
        try {
            out.write(TotalContenByte);
        }
        catch (Exception e) {
            logger.error("WriteHDFS_Error:\t" + e.toString());
            SuccOrNot = false;
        } finally {
            out.close();
            return SuccOrNot;
        }

    }

//    public static boolean Write(String[] ContentArr,String DesPath) throws IOException {
//        boolean SuccOrNot = true;
//        logger = Logger.getLogger(ReadNWrite.class.getName());
//        FileWriter fos = new FileWriter(DesPath);
//        BufferedWriter bw = new BufferedWriter(fos);
//        try {
//            for (int i = 0; i < ContentArr.length; i++) {
//                bw.write(ContentArr[i] + "\n");
//            }
//        } catch (Exception e) {
//            logger.error("WriteHDFS_Error:\t" + e.toString());
//            SuccOrNot = false;
//        } finally {
//            bw.close();
//            return SuccOrNot;
//        }
//    }
}
