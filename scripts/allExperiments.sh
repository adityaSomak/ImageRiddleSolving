### Completed in GPU Machine 1 ###
#python testAllStagesIUR.py intermediateFiles/lemmatizedSeedsCentralized_all3k.txt \ 
#../First250riddles/riddlesDetectionAll/ 3333 intermediateFiles/output_iur_all_pc1/ all

### Completed My Machine ###
#python testAllStagesIUR.py intermediateFiles/lemmatizedSeedsCentralized_all3k.txt \ 
#../First250riddles/riddlesDetectionAll/ 3333 intermediateFiles/output_iur_merge_pc1/ merge

### Completed My Machine ###
#python testAllStagesIUR.py intermediateFiles/lemmatizedSeedsCentralized_all3k.txt \ 
#../First250riddles/riddlesDetectionAll/ 3333 intermediateFiles/output_iur_clarifai_pc1/ clarifai

### Completed on GPU Machine 1 ###
#python testAllStagesGUR.py intermediateFiles/lemmatizedSeedsCentralized_all3k.txt \ 
#../First250riddles/riddlesDetectionAll/ 3333 intermediateFiles/output_gur_all_pc1/ all

### Completed My Machine ###
#python testAllStagesGUR.py intermediateFiles/lemmatizedSeedsCentralized_all3k.txt \ 
#../First250riddles/riddlesDetectionAll/ 3333 intermediateFiles/output_gur_merge_pc1/ merge

### Completed My Machine ###
#python testAllStagesGUR.py intermediateFiles/lemmatizedSeedsCentralized_all3k.txt \ 
#../First250riddles/riddlesDetectionAll/ 3333 intermediateFiles/output_gur_clarifai_pc1/ clarifai

python testAllStagesUR.py intermediateFiles/lemmatizedSeedsCentralized_all3k.txt \ 
../First250riddles/riddlesDetectionAll/ 3333 intermediateFiles/output_ur_all_pc1/ all

### Completed My Machine ###
#python testAllStagesUR.py intermediateFiles/lemmatizedSeedsCentralized_all3k.txt \ 
#../First250riddles/riddlesDetectionAll/ 3333 intermediateFiles/output_ur_merge_pc1/ merge

### Completed My Machine ###
#python testAllStagesUR.py intermediateFiles/lemmatizedSeedsCentralized_all3k.txt \ 
#../First250riddles/riddlesDetectionAll/ 3333 intermediateFiles/output_ur_clarifai_pc1/ clarifai

#-----------------------------
python amtCSVCreator.py intermediateFiles/output_gur_all_pc2n/ \
intermediateFiles/output_ur_clarifai_pc1/ intermediateFiles/resnet/output_ur_clarifai_pc1/
#-------------------------------
Type in R:
----------
amt = read.csv("amtCSV_results.csv")
jpeg('amt_correctness_1.jpg');
barplot(matrix(c(table(amt$Answer.Q3Answer), table(amt$Answer.Q5Answer), table(amt$Answer.Q7Answer)),nr=3, byrow=TRUE),beside=T,col=c("dodgerblue4","brown3","cornsilk"),names.arg=c(1,2,3,4,5,6),axes=FALSE,main="Correctness")
legend("topright",c("GUR","Clarifai","ResidualNet"),col=c("dodgerblue4","brown3","cornsilk"),pch=15,pt.cex=3);
labs <- c(0,550,1100,1650,2200); axis(side = 2, at = labs, labels = paste0(ceiling(labs / 22), "%"), cex.axis = 0.7)
dev.off()

jpeg('amt_explainability_1.jpg');
barplot(matrix(c(table(amt$Answer.Q4Answer), table(amt$Answer.Q6Answer), table(amt$Answer.Q8Answer)),nr=3, byrow=TRUE),beside=T,col=c("dodgerblue4","brown3","cornsilk"),names.arg=c(1,2,3,4),axes=FALSE, main="Intelligence")
legend("topright",c("GUR","Clarifai","ResidualNet"),col=c("dodgerblue4","brown3","cornsilk"),pch=15,pt.cex=3);
labs <- c(0,675,1350,2025,2700); axis(side = 2, at = labs, labels = paste0(ceiling(labs / 27), "%"), cex.axis = 0.7)
dev.off()

-----------------------------------
barplot(matrix(c(table(amt$Answer.Q4Answer), table(amt$Answer.Q6Answer), table(amt$Answer.Q8Answer,exclude=1.75)),nr=3, byrow=TRUE),beside=T,col=c("dodgerblue4","brown3","cornsilk"),names.arg=c(1,1.5,2,2.5,3,3.5,4))

barplot(matrix(c(table(amt$Answer.Q3Answer,exclude=1.75), table(amt$Answer.Q5Answer,exclude=1.75), table(amt$Answer.Q7Answer)),nr=3, byrow=TRUE),beside=T,col=c("dodgerblue4","brown3","cornsilk"),names.arg=c(1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6))
