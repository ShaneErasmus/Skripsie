#Take pictures [1]
python3 TakePics.py

#Image Registration [2]
python3 Image_Registration.py

#Image Segementation [3]
#-----------------TOP IMAGE-------------------------
echo "Processing Top Image..."
# Fiji initial
java -classpath "../Fiji.app/jars/*:."  Segmentation "Images/T.jpg"
# Convex python 
python3 Convex.py
#Fiji final clean
java -classpath "../Fiji.app/jars/*:." Clean "T"

#------------------BOTTOM IMAGE---------------------
echo "Processing Bottom Image..."
# Fiji initial
java -classpath "../Fiji.app/jars/*:." Segmentation "Images/B.jpg"
# Convex python 
python3 Convex.py
# Fiji final clean
java -classpath "../Fiji.app/jars/*:." Clean "B"

# Evalute particles [4]
python3 Processing.py

#Delete temp files [5]
echo "Cleaning unwanted files..."
rm "../Convex.png"
rm "../ResultMask.png"






















