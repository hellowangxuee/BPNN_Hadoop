package HDFS_IO;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.IOUtils;
import org.apache.log4j.Logger;
import org.apache.hadoop.fs.Path;
import sun.security.krb5.internal.crypto.Des;

import java.io.IOException;
import java.net.URI;

/**
 * Created by mlx on 3/17/16.
 */
public class ReadNWrite {
    static Logger logger = null;

    public static String[] hdfs_Read(String filePath) throws IOException {
        String[] ContentArr = null;

        logger = Logger.getLogger(ReadNWrite.class.getName());

        Configuration conf = new Configuration();
        conf.set("hadoop.job.ugi", "hadoop-user,hadoop-user");
        //FileSystem是用户操作HDFS的核心类，它获得URI对应的HDFS文件系统
        FileSystem fs = FileSystem.get(URI.create(filePath), conf);
        FSDataInputStream in = null;
        String contentStr = "";
        try {
            in = fs.open(new Path(filePath));
            byte[] ContentBuff = new byte[4096 * 10];
            in.read(ContentBuff);
            for (byte b : ContentBuff) {
                if (b != 0) {
                    contentStr += (char) b;
                }
            }
            ContentArr = contentStr.split("\n");
        } catch (Exception e) {
            logger.error("ReadHDFS_Error:\t" + e.toString());
        } finally {
            IOUtils.closeStream(in);
            return ContentArr;
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
        } catch (Exception e) {
            logger.error("WriteHDFS_Error:\t" + e.toString());
            SuccOrNot = false;
        } finally {
            out.close();
            return SuccOrNot;
        }

    }
}
