#!/usr/bin/python
import random
from os import listdir
from os.path import isfile, join
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys, getopt


def main(argv):
	try:
		opts, args = getopt.getopt(argv,"hn:",["nfile="])
	except getopt.GetoptError:
		print 'test.py -n <outputfilename>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'test.py -i <inputfile> -o <outputfile>'
			sys.exit()
		elif opt in ("-n", "--nfile"):
			nfile = arg
	global keyev
	with open('outfileok/'+nfile+'_out.txt', 'w') as outfile:
		keyev=False
		def press(event):
			print('press', event.key)
			keyev=True
			k = event.key	
			outfile.write(fname+' '+str(time.time())+' '+k+'\n')
			plt.close()

		onlyfiles = [f for f in listdir("img") if isfile(join("img", f))]

		random.shuffle(onlyfiles)
		for fname in onlyfiles:	
				fig, ax = plt.subplots()

				fig.canvas.mpl_connect('key_press_event', press)
		
				#print fname
				img=mpimg.imread("img/"+fname,0)
				plt.imshow(img)
				mng = plt.get_current_fig_manager()
				mng.resize(*mng.window.maxsize())
				plt.show(block=True)
				keyev=False
	print("-------------------")
	print("Thank you for your participation!") 
if __name__ == "__main__":
	main(sys.argv[1:])
