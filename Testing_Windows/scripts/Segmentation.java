import ij.IJ;
import ij.ImagePlus;
import ij.plugin.ImageCalculator;
import ij.plugin.frame.RoiManager;
import ij.process.ImageProcessor;
import ij.io.FileSaver;

public class Segmentation   {

    public static void main(String[] args) {
        System.out.println("Segmentation algorithm running...");
        String imagePath = "\\"  + args[0];                             //*******************CHANGE THIS TO THE PATH OF THE T.jpg and B.jpg**********************
        ImagePlus imp = IJ.openImage(imagePath);
        // Set the initial thresholds for each channel (Hue, Saturation, and Brightness)
        int hMin = 0;
        int hMax = 255;
        int sMin = 0;
        int sMax = 255;
        int bMin = 35;
        int bMax = 255;

        // Convert the image to HSB
        IJ.run(imp, "HSB Stack", "");
        IJ.run(imp, "Convert Stack to Images", "");

        // Apply threshold to each channel
        ImageProcessor hueProcessor = imp.getImageStack().getProcessor(1);
        hueProcessor.setThreshold(hMin, hMax, ImageProcessor.NO_LUT_UPDATE);
        ImagePlus hueMask = new ImagePlus("Hue_Mask", hueProcessor.createMask());

        ImageProcessor saturationProcessor = imp.getImageStack().getProcessor(2);
        saturationProcessor.setThreshold(sMin, sMax, ImageProcessor.NO_LUT_UPDATE);
        ImagePlus saturationMask = new ImagePlus("Saturation_Mask", saturationProcessor.createMask());

        ImageProcessor brightnessProcessor = imp.getImageStack().getProcessor(3);
        brightnessProcessor.setThreshold(bMin, bMax, ImageProcessor.NO_LUT_UPDATE);
        ImagePlus brightnessMask = new ImagePlus("Brightness_Mask", brightnessProcessor.createMask());

        //Combine the masks
        ImageCalculator ic = new ImageCalculator();
        ImagePlus resultMask = ic.run("AND create", hueMask, saturationMask);
        resultMask = ic.run("AND create", resultMask, brightnessMask);
        resultMask.setTitle("Result_Mask");

        //Clear intermediate masks
        hueMask.close();
        saturationMask.close();
        brightnessMask.close();

        //Process the result mask
        // IJ.selectWindow("Result_Mask");
        IJ.run(resultMask, "8-bit", "");
        IJ.run(resultMask,"Erode", "");
        IJ.run(resultMask, "Watershed", "");

        // Save the resultMask as PNG
        FileSaver fs = new FileSaver(resultMask);
        String outputImagePath = "C:/Users/Shane/Documents/Universiteit/Fourth Year/Skripsie/ResultMask.png"; //***************CHANGE THIS TO DESIRED OUTPUT PATH*******************
        fs.saveAsPng(outputImagePath);
    }
        
}










