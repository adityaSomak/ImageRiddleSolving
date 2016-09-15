import matplotlib.image as mpimg
import pylab
import os

workdir=os.path.join("../../First250riddles/riddlesDetectionAll/Image/")

#outdir = os.path.join("./figureOut2/")
#f = open('summaryFile.txt','r')
outdir = os.path.join("./figureOutNeg/")
f = open('negativeSummaryFile.txt','r')
i=0;
for line in f:
    print line
    tmp = line.strip().split();
    if i==0 or tmp[2].isupper():
        i=i+1;
        continue;
    file=tmp[0] + '_1.png'
    complete_path=os.path.join(workdir,file)
    i1=mpimg.imread(complete_path)
    file=tmp[0] + '_2.png'
    complete_path=os.path.join(workdir,file)
    i2=mpimg.imread(complete_path)
    file=tmp[0] + '_3.png'
    complete_path=os.path.join(workdir,file)
    i3=mpimg.imread(complete_path)
    file=tmp[0] + '_4.png'
    complete_path=os.path.join(workdir,file)
    i4=mpimg.imread(complete_path)
    
    pylab.subplot(1,4,1,frameon=False, xticks=[], yticks=[])
    pylab.imshow(i1) 
    pylab.subplot(1,4,2,frameon=False, xticks=[], yticks=[])
    pylab.imshow(i2)
    pylab.subplot(1,4,3,frameon=False, xticks=[], yticks=[])
    pylab.imshow(i3)
    pylab.subplot(1,4,4,frameon=False, xticks=[], yticks=[])
    pylab.imshow(i4)
    #pylab.show() 
    pylab.title('GroundTruth: %s\n GUR: %s, Baseline: %s' % (tmp[0], tmp[1], tmp[2]), loc='right', fontsize=20)
    pylab.savefig(outdir+tmp[0], bbox_inches='tight')
    i=i+1;

f.close()
