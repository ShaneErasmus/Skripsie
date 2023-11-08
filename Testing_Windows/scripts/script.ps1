#RUN THIS SCRIPT THAT DOES IMAGE PROCESSING
#CHANGE THE PATHS SO THAT THEY MATCH THE "fiji-win32" FILE PATH
#ALSO CHANGE PATHS SO THAT ARE IN MAIN

#-----------------TOP IMAGE-------------------------
Write-Host "-----------------------------"
Write-Host "Processing Top Image..."
Write-Host "-----------------------------"
#Fiji initial
java -classpath "C:\Users\Shane\Documents\Universiteit\Fourth Year\Skripsie\Testing_Windows\fiji-win32\Fiji.app\jars\*;." Segmentation "T.jpg"
#Convex python 
python Convex.py
#Fiji final clean
java -classpath "C:\Users\Shane\Documents\Universiteit\Fourth Year\Skripsie\Testing_Windows\fiji-win32\Fiji.app\jars\*;." Clean "T"


#------------------BOTTOM IMAGE---------------------
Write-Host "-----------------------------"
Write-Host "Processing Bottom Image..."
Write-Host "-----------------------------"
#Fiji initial
java -classpath "C:\Users\Shane\Documents\Universiteit\Fourth Year\Skripsie\Testing_Windows\fiji-win32\Fiji.app\jars\*;." Segmentation "B.jpg"
#Convex python 
python Convex.py
#Fiji final clean
java -classpath "C:\Users\Shane\Documents\Universiteit\Fourth Year\Skripsie\Testing_Windows\fiji-win32\Fiji.app\jars\*;." Clean "B"

#Save particles python
Write-Host "-----------------------------"
python Processing.py


#-----------Delete temp files---------------------
Write-Host "Cleaning unwanted files..."
Remove-Item "C:\Users\Shane\Documents\Universiteit\Fourth Year\Skripsie\Testing_Windows\Convex.png" -Force
Remove-Item "C:\Users\Shane\Documents\Universiteit\Fourth Year\Skripsie\Testing_Windows\ResultMask.png" -Force
