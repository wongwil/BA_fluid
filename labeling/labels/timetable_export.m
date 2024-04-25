clear
filename = "labels/vial33_labels";
load(filename + ".mat")

labelData = gTruth.LabelData;
writetimetable(labelData,filename+".csv")
