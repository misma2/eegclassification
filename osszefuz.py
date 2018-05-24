#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys, getopt
import csv
import numpy
import matplotlib.pyplot as plt

def main(argv):
        try:
                opts, args=getopt.getopt(argv,"hn:e:k:fls",["nfile=","eegfile=","keys="])
        except getopt.GetoptError:
                print 'test.py -n <nickname> -e <eegfile> -k <keys> '
		print '-f for the first measurement'
		print '-l for the last measurement'
		print '-s for the second measurement'
                sys.exit(2)
	first=False;
	last=False;
	second=False;
        for opt, arg in opts:
                if opt == '-h':
	                print 'test.py -n <nickname> -e <eegfile> -k <keys> '
			print '-f for the first measurement'
			print '-s for the second measurement'
			print '-l for the last measurement'
                        sys.exit()
                elif opt in ("-n", "--nfile"):
                        nfile = arg
                elif opt in ("-e", "--eegfile"):
                        efile = arg
		elif opt in ("-k","--keys"):
			keys = arg
		elif opt == '-f':
			first=True;
		elif opt == '-l':
			last=True;
		elif opt == '-s':
			second=True;

	reader = csv.reader(open(efile,"rb"), delimiter=",")
	x = list(reader)
	x.pop(0)
	for row in x:
		del row[len(row)-1]
		del row[len(row)-1]
		del row[3]
		del row[4]
		del row[5]
		del row[6]
	eegresult = numpy.array(x).astype("float")
	reader2 = csv.reader(open("../script/outfileok/"+nfile+"_out.txt", "rb"),delimiter=" ")
	x2=list(reader2)
	for row in x2:
		if row[0]=="Képernyőkép":
			del row[3]
			del row[2]
			del row[1]
		if row[0]=="Róma":
			del row[1]
		del row[0]
		if row[1]==keys[0]:
			row[1]=-1
		if row[1]==keys[1]:
			row[1]=0
		if row[1]==keys[2]:
			row[1]=1
	imgresult = numpy.array(x2).astype("float")
	imgresult[:,0]*=1000
	x3=[]
	zerogoodness=[]
	os.system('mkdir plots/'+nfile)
	goodness=[]
	#######Innen kell csinalni a ciklust #########
	for k in range(2,10):
		j=0
		for i in range (0,len(imgresult)):
			temp=[]
			while imgresult[i,0]>eegresult[j,0]:
				temp.append(eegresult[j,k]) #itt
				j+=1;
			tempi=numpy.array(temp)
			x3.append(tempi)
		splittedeeg=numpy.array(x3)
		os.system('mkdir plots/'+nfile+'/'+str(k))
		i=0
		for i in range(0,len(splittedeeg)):
			plt.plot(numpy.arange(0, len(splittedeeg[i])), splittedeeg[i])
			plt.savefig('plots/'+nfile+'/'+str(k)+'/'+str(i)+'.png')
			plt.close()
		zerocrossings1=[]
		zerocrossings0=[]
		zerocrossings_1=[]
		originalzerocrossings=[]
		taverages_1=[]
		taverages0=[]
		taverages1=[]
		i=0
		originalaverages=[]
		for i in range(0,len(splittedeeg)):
	#		print(len(splittedeeg))
			#zerocrossings
			nozero=splittedeeg[i]
			nozero[nozero==0]=-1
			zerocrossing=numpy.count_nonzero(numpy.diff(numpy.sign(nozero)))*100/len(splittedeeg[i])

			#zerocrossing=len(numpy.where(numpy.diff(nozero))[0])/len(splittedeeg[i])
			originalzerocrossings.append(zerocrossing)
			#integralfeature
			average=numpy.average(numpy.absolute(splittedeeg[i]))
			originalaverages.append(average)
			#put them to the right class
			if imgresult[i,1]==-1:
				zerocrossings_1.append(zerocrossing)
				taverages_1.append(average)
			elif imgresult[i,1]==0:
				zerocrossings0.append(zerocrossing)
				taverages0.append(average)
			else:
				zerocrossings1.append(zerocrossing)
				taverages1.append(average)
		originalzerocrossings=numpy.array(originalzerocrossings)
		originalaverages=numpy.array(originalaverages)
	#	allaverages=[]
		zerocrossings_1=numpy.array(zerocrossings_1)
		zerocrossings0=numpy.array(zerocrossings0)
		zerocrossings1=numpy.array(zerocrossings1)


		averages_1=numpy.array(taverages_1)
		averages0=numpy.array(taverages0)
		averages1=numpy.array(taverages1)

	#	allaverages.append(averages_1)
	#	allaverages.append(averages0)
	#	allaverages.append(averages1)

	#	allaverages=numpy.array(allaverages)
		zeroaver_1=numpy.average(zerocrossings_1)
		zeroaver0=numpy.average(zerocrossings0)
		zeroaver1=numpy.average(zerocrossings1)

		aver_1=numpy.average(averages_1)
		aver0=numpy.average(averages0)
		aver1=numpy.average(averages1)


		zerosameclasses=[]
		sameclasses=[]
		i=0
		j=0
		#print(allaverages[0,:])
		for i in range(0,len(originalaverages)):
			zerotemp=[]
			temp=[]
			zerod1=originalzerocrossings[i]-zeroaver_1
			zerod2=originalzerocrossings[i]-zeroaver0
			zerod3=originalzerocrossings[i]-zeroaver1
			d1=originalaverages[i]-aver_1
			d2=originalaverages[i]-aver0
			d3=originalaverages[i]-aver1
			zerotemp.append(zerod1)
			zerotemp.append(zerod2)
			zerotemp.append(zerod3)
			temp.append(d1)
			temp.append(d2)
			temp.append(d3)
			zerotemp=numpy.absolute(numpy.array(zerotemp))
			temp=numpy.absolute(numpy.array(temp))
			if numpy.argmin(zerotemp)-1==imgresult[i,1]:
				zerosameclasses.append(1)
			else:
				zerosameclasses.append(0)
			if numpy.argmin(temp)-1==imgresult[i,1]:
				sameclasses.append(1)
			else:
				sameclasses.append(0)
		zerosameclasses=numpy.array(zerosameclasses)
		sameclasses=numpy.array(sameclasses)
		zeroresult=numpy.average(zerosameclasses)*100
		result=numpy.average(sameclasses)*100
		print(k)
		print(zeroresult)
		print(result) ####ezek minden ciklus eredmnyei
		zerogoodness.append(zeroresult)
		goodness.append(result)
		x3=[]
		if not first:
			eegbefore=numpy.load('1eeg'+str(k)+'.npy');
			eegs=numpy.concatenate((eegbefore,splittedeeg),axis=0);
			print(len(eegbefore))
			print(len(eegs))
			numpy.save('1eeg'+str(k),eegs) #elsonek splittedeeg
		else:
			numpy.save('1eeg'+str(k),splittedeeg) #elsonek splittedeeg

	############################
	zerogoodness=numpy.array(zerogoodness).astype('float')
	if not first:
		zerogoodnessbefore=numpy.load('zerogoodness1.npy')
		if second:
			zerogoodnessafter=numpy.append([zerogoodnessbefore],[zerogoodness],axis=0)
		else:
			zerogoodnessafter=numpy.append(zerogoodnessbefore,[zerogoodness],axis=0)
		numpy.save('zerogoodness1',zerogoodnessafter)
	else:
		numpy.save('zerogoodness1',zerogoodness)		 
#zerogoodnessafter);
	goodness=numpy.array(goodness).astype('float')
	if not first: 
		goodnessbefore=numpy.load('goodness1.npy')
		if second:
			goodnessafter=numpy.append([goodnessbefore],[goodness],axis=0) #elsonek kell zarojel a beforenak is
		else:
			goodnessafter=numpy.append(goodnessbefore,[goodness],axis=0) #elsonek kell zarojel a beforenak is

		numpy.save('goodness1',goodnessafter);
	else:
		numpy.save('goodness1',goodness);		
	if not first:
		imgbefore=numpy.load('imgresult1.npy');
		imgafter=numpy.concatenate((imgbefore,imgresult),axis=0)
		numpy.save('imgresult1',imgafter) #elsonek imgresult
	else:
		numpy.save('imgresult1',imgresult) #elsonek imgresult
	if last:
		numpy.savetxt("zerogoodness1.csv", zerogoodnessafter, delimiter=",")
		numpy.savetxt("intgoodness1.csv", goodnessafter, delimiter=",")
if __name__ == "__main__":
        main(sys.argv[1:])
