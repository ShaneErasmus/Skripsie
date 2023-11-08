import ij.IJ;
import ij.ImagePlus;
import ij.io.FileSaver;

public class Clean   {

    public static void main(String[] args) {
        System.out.println("Final cleaning...");
        String imagePath = "../Convex.png";
        ImagePlus imp = IJ.openImage(imagePath);
    
        IJ.run(imp, "8-bit", "");
        IJ.run(imp, "Watershed", "");

        FileSaver fs = new FileSaver(imp);
        String outputImagePath = "../FinalMask" + args[0] + ".png";
        fs.saveAsPng(outputImagePath);
    }
        
}










