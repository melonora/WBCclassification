from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from scipy import misc
import torch.nn as nn
import numpy as np
import torch
import os
import random
import glob
from random import randint as ri
import cv2
from funcs import *
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle as cp

class ResBlock(nn.Module):
	# Actual CNN
	def __init__(s, hidden=16):
		super(ResBlock, s).__init__()
		s.conv1 = nn.Conv2d(3, hidden, kernel_size=3, padding=1)
		s.conv2 = nn.Conv2d(3, hidden, kernel_size=3, padding=3, dilation=3)
		s.bn1 = nn.ModuleList([nn.BatchNorm2d(hidden)for i in xrange(5)])
		s.bn2 = nn.ModuleList([nn.BatchNorm2d(hidden)for i in xrange(5)])
		s.conv3 = nn.Conv2d(hidden*2, 3, kernel_size=3, padding=3, dilation=3)

	def resblock(s,x,i):
		out1 = F.relu(s.bn1[i](s.conv1(x)))
		out2 = F.relu(s.bn2[i](s.conv2(x)))
		out = s.conv3(torch.cat((out1, out2), 1))
		out += x
		return out

	def forward(s,x, save=False):
		for i in xrange(5):
			if save: misc.imsave('%i.png'%i,torch2misc(x[0]))
			x = s.resblock(x,i)
		if save:misc.imsave('%i.png'%(5),torch2misc(x[0]))
		return F.softmax(x, dim = 1)
		
def testFoldLoss(o1,o2,o3,o4,o5,l1,l2,l3,l4,l5,foldLsL):
	#Obtaining the MSE loss over the stopping bins of each fold and storing the result in a list.
	tmpL1 = criterion(o1[12:],l1[12:]).data.cpu().numpy()
	tmpL2 = criterion(o2[11:],l2[11:]).data.cpu().numpy()
	tmpL3 = criterion(o3[8:],l3[8:]).data.cpu().numpy()
	tmpL4 = criterion(o4[12:],l4[12:]).data.cpu().numpy()
	tmpL5 = criterion(o5[10:],l5[10:]).data.cpu().numpy()

	foldLsL.append([tmpL1,tmpL2,tmpL3,tmpL4,tmpL5,sum([tmpL1,tmpL2,tmpL3,tmpL4,tmpL5])])
	return foldLsL, sum([tmpL1,tmpL2,tmpL3,tmpL4,tmpL5])

def valFoldLoss(o1,o2,o3,o4,o5,l1,l2,l3,l4,l5,foldLsL):
	#Obtaining the MSE loss over the validation bins of each fold and storing the result in a list.
	tmpL1 = criterion(o1[:12],l1[:12]).data.cpu().numpy()
	tmpL2 = criterion(o2[:11],l2[:11]).data.cpu().numpy()
	tmpL3 = criterion(o3[:8],l3[:8]).data.cpu().numpy()
	tmpL4 = criterion(o4[:12],l4[:12]).data.cpu().numpy()
	tmpL5 = criterion(o5[:10],l5[:10]).data.cpu().numpy()

	foldLsL.append([tmpL1,tmpL2,tmpL3,tmpL4,tmpL5,sum([tmpL1,tmpL2,tmpL3,tmpL4,tmpL5])])
	return foldLsL, sum([tmpL1,tmpL2,tmpL3,tmpL4,tmpL5])
	
def nextPath(path1,path2,path3,path4,path5):
	#Rearranging the paths to each validation single cell image to obtain the list of paths to validation images for the next fold.
	path1= path1[12:]+path1[:12]
	path2= path2[11:]+path1[:11]
	path3= path3[8:]+path1[:8]
	path4= path4[12:]+path1[:12]
	path5= path5[10:]+path1[:10]
	return path1,path2,path3,path4,path5

def next(o1,o2,o3,o4,o5,l1,l2,l3,l4,l5):
	#Rearranging the output of the network so they are in the correct order for the next fold.
	o1 = torch.cat((o1[12:],o1[:12]),0)
	o2 = torch.cat((o2[11:],o2[:11]),0)
	o3 = torch.cat((o3[8:],o3[:8]),0)
	o4 = torch.cat((o4[12:],o4[:12]),0)
	o5 = torch.cat((o5[10:],o5[:10]),0)
	
	#Rearranging the labels corresponding to the output of the network so they are in the correct order for the next fold.
	l1 = torch.cat((l1[12:],l1[:12]),0)
	l2 = torch.cat((l2[11:],l2[:11]),0)
	l3 = torch.cat((l3[8:],l3[:8]),0)
	l4 = torch.cat((l4[12:],l4[:12]),0)
	l5 = torch.cat((l5[10:],l5[:10]),0)
	return o1,o2,o3,o4,o5,l1,l2,l3,l4,l5
	
def makeValImg(output,cell,sDir,valPath):
	# Make an image with both the real cell image and the segmentation output.
	empty = np.zeros((len(valPath),120,120,3))
	for i in xrange(3):
		empty[:,:,:,i] = output[:,i,:,:].cpu().data.numpy()
	for i in xrange(len(empty)):
		img = np.zeros((120,120,3))
		mi = np.argmax(empty,axis=3)
		for w in xrange(120):
			for s in xrange(120):
				img[w,s,mi[i,w,s]] = 1
		misc.imsave(modPath+sDir+'/'+cell+"%i.png"%i,misc.imresize(np.concatenate((img,misc.imread(valPath[i])/255.)),200))
	
def train(epoch,criterion,vF1,vF2,vF3,vF4,vF5,batchSize = 60):
	# Defining basic initial parameters and lists to store results in.
	cBatchSize = batchSize/5
	train_loss = 0
	trainLossTemp = 0
	batchIter = []
	batchIter2 = []
	trainLossIter = []
	testLossIter = []
	fold1LsL = []
	fold2LsL = []
	fold3LsL = []
	fold4LsL = []
	fold5LsL = []
	val1Ls = []
	val2Ls = []
	val3Ls = []
	val4Ls = []
	val5Ls = []
	valLsTot = []
	
	# Loading the validation data
	valData1 = Variable(torch.cat(tuple(misc2torch(misc.imread(i)[:,:,:3]/255.).view(1,3,120,120) for i in neutV),0).float()).to(device)
	valData2 = Variable(torch.cat(tuple(misc2torch(misc.imread(i)[:,:,:3]/255.).view(1,3,120,120) for i in basoV),0).float()).to(device)
	valData3 = Variable(torch.cat(tuple(misc2torch(misc.imread(i)[:,:,:3]/255.).view(1,3,120,120) for i in eosiV),0).float()).to(device)
	valData4 = Variable(torch.cat(tuple(misc2torch(misc.imread(i)[:,:,:3]/255.).view(1,3,120,120) for i in lymphoV),0).float()).to(device)
	valData5 = Variable(torch.cat(tuple(misc2torch(misc.imread(i)[:,:,:3]/255.).view(1,3,120,120) for i in monoV),0).float()).to(device)
	
	# Create list of indices used to load in the synthetic training data.
	nindex = range(80000)
	random.shuffle(nindex)
	
	# For loop by which the CNN is trained
	for batch in xrange(0,80000,batchSize):
		print batch
		model.train()
		optimizer.zero_grad()
		
		#Create sublist in order to obtain the indices used to load in the synthetic image data for the specific batch. Synthetic training data is loaded afterwards. Augmentation can be adjusted in funcs.py.
		ls = [i for i in nindex[batch:batch+batchSize]]
		data1 = torch.cat(tuple(misc2torch(augment(misc.imread(tPath1+str(i)+".png")[:,:,:3]/255.)).view(1,3,120,120) for i in ls),0).float()
		data2 = torch.cat(tuple(misc2torch(augment(misc.imread(tPath2+str(i)+".png")[:,:,:3]/255.)).view(1,3,120,120) for i in ls),0).float()
		data3 = torch.cat(tuple(misc2torch(augment(misc.imread(tPath3+str(i)+".png")[:,:,:3]/255.)).view(1,3,120,120) for i in ls),0).float()
		data4 = torch.cat(tuple(misc2torch(augment(misc.imread(tPath4+str(i)+".png")[:,:,:3]/255.)).view(1,3,120,120) for i in ls),0).float()
		data5 = torch.cat(tuple(misc2torch(augment(misc.imread(tPath5+str(i)+".png")[:,:,:3]/255.)).view(1,3,120,120) for i in ls),0).float()
		data = Variable(torch.cat((data1,data2,data3,data4,data5),0)).to(device)
		
		label1 = torch.cat(tuple(mask2label(misc.imread(lPath1+str(i)+".png")).view(1,3,120,120) for i in ls),0).float()
		label2 = torch.cat(tuple(mask2label(misc.imread(lPath2+str(i)+".png")).view(1,3,120,120) for i in ls),0).float()
		label3 = torch.cat(tuple(mask2label(misc.imread(lPath3+str(i)+".png")).view(1,3,120,120) for i in ls),0).float()
		label4 = torch.cat(tuple(mask2label(misc.imread(lPath4+str(i)+".png")).view(1,3,120,120) for i in ls),0).float()
		label5 = torch.cat(tuple(mask2label(misc.imread(lPath5+str(i)+".png")).view(1,3,120,120) for i in ls),0).float()
		label = Variable(torch.cat((label1,label2,label3,label4,label5),0)).to(device)
		
		#Feeding the synthetic images to the CNN. Determining the loss. Performing back propagation and updating the model parameters.
		output = model(data)
		loss = criterion(output,label)
		loss.backward()
		optimizer.step()
		train_loss += loss.data
		del data,label
		
		# Checking performance on real cell images every 1800 images.
		if batch % 360 == 0 and batch != 0:
			trainLossIter.append((train_loss.cpu().numpy()-trainLossTemp)/30.)
			batchIter.append(batch*5)
			trainLossTemp = train_loss.cpu().numpy()
			model.eval()
			
			#Obtaining indices used for loading in synthetic test data, after which this is loaded in alongside the corresponding labels.
			nindex2 = [ri(80000,99999) for i in xrange(20)]
			testNeut = torch.cat(tuple(misc2torch(misc.imread(tPath1+str(i)+".png")[:,:,:3]/255.).view(1,3,120,120) for i in nindex2),0).float()
			testBaso = torch.cat(tuple(misc2torch(misc.imread(tPath2+str(i)+".png")[:,:,:3]/255.).view(1,3,120,120) for i in nindex2),0).float()
			testEosi = torch.cat(tuple(misc2torch(misc.imread(tPath3+str(i)+".png")[:,:,:3]/255.).view(1,3,120,120) for i in nindex2),0).float()
			testLympho = torch.cat(tuple(misc2torch(misc.imread(tPath4+str(i)+".png")[:,:,:3]/255.).view(1,3,120,120) for i in nindex2),0).float()
			testMono = torch.cat(tuple(misc2torch(misc.imread(tPath5+str(i)+".png")[:,:,:3]/255.).view(1,3,120,120) for i in nindex2),0).float()
			tData = Variable(torch.cat((testNeut,testBaso,testEosi,testLympho,testMono),0)).to(device)

			tLabel1 = torch.cat(tuple(mask2label(misc.imread(lPath1+str(i)+".png")).view(1,3,120,120) for i in nindex2),0).float()
			tLabel2 = torch.cat(tuple(mask2label(misc.imread(lPath2+str(i)+".png")).view(1,3,120,120) for i in nindex2),0).float()
			tLabel3 = torch.cat(tuple(mask2label(misc.imread(lPath3+str(i)+".png")).view(1,3,120,120) for i in nindex2),0).float()
			tLabel4 = torch.cat(tuple(mask2label(misc.imread(lPath4+str(i)+".png")).view(1,3,120,120) for i in nindex2),0).float()
			tLabel5 = torch.cat(tuple(mask2label(misc.imread(lPath5+str(i)+".png")).view(1,3,120,120) for i in nindex2),0).float()
			tLabels = Variable(torch.cat((tLabel1,tLabel2,tLabel3,tLabel4,tLabel5),0)).to(device)
			
			# Loading the labels corresponding to the validation image data.
			valLabel1 = Variable(torch.cat(tuple(mask2label(misc.imread(i)).view(1,3,120,120) for i in neutVLab),0).float()).to(device)
			valLabel2 = Variable(torch.cat(tuple(mask2label(misc.imread(i)).view(1,3,120,120) for i in basoVLab),0).float()).to(device)
			valLabel3 = Variable(torch.cat(tuple(mask2label(misc.imread(i)).view(1,3,120,120) for i in eosiVLab),0).float()).to(device)
			valLabel4 = Variable(torch.cat(tuple(mask2label(misc.imread(i)).view(1,3,120,120) for i in lymphoVLab),0).float()).to(device)
			valLabel5 = Variable(torch.cat(tuple(mask2label(misc.imread(i)).view(1,3,120,120) for i in monoVLab),0).float()).to(device)
			
			# Feed synthetic test data to the model and calculate loss which is stored in a list.
			with torch.no_grad():
				output = model(tData)
				loss = criterion(output,tLabels)
				tLoss = loss.data.cpu().numpy()
				batchIter2.append(batch*5)
				testLossIter.append(tLoss)
				del tData,tLabels
			# Feed real cell images to the model
			with torch.no_grad():
				output1 = model(valData1)
				output2 = model(valData2)
				output3 = model(valData3)
				output4 = model(valData4)
				output5 = model(valData5)
			
			# Determine loss for each validation image of the stopping bin of each fold. Compare this to the previous lowest loss. If lower determine loss of validation images for the specific fold. In this case of each validation image an image is created with the validation image alongside the segmentation output. The model is saved and the output and labels are rearranged so they are in correct order for the next fold.	
			fold1LsL,tmpF1 = testFoldLoss(output1,output2,output3,output4,output5,valLabel1,valLabel2,valLabel3,valLabel4,valLabel5,fold1LsL)
			if tmpF1 < fLoss[0]:
				val1Ls, vF1 = valFoldLoss(output1,output2,output3,output4,output5,valLabel1,valLabel2,valLabel3,valLabel4,valLabel5,val1Ls)
				fLoss[0] = tmpF1
				makeValImg(output1[:12],'neut',subDir2[0],neutV[:12])
				makeValImg(output2[:11],'baso',subDir2[0],basoV[:11])
				makeValImg(output3[:8],'eosi',subDir2[0],eosiV[:8])
				makeValImg(output4[:12],'lymfo',subDir2[0],lymphoV[:12])
				makeValImg(output5[:10],'mono',subDir2[0],monoV[:10])
				torch.save(model, modPath + "/fold1/" +str(batch*5)+"_"+str(tmpF1))
			output1,output2,output3,output4,output5,valLabel1,valLabel2,valLabel3,valLabel4,valLabel5 = next(output1,output2,output3,output4,output5,valLabel1,valLabel2,valLabel3,valLabel4,valLabel5)
			
			fold2LsL,tmpF2 = testFoldLoss(output1,output2,output3,output4,output5,valLabel1,valLabel2,valLabel3,valLabel4,valLabel5,fold2LsL)
			if tmpF2 < fLoss[1]:
				val2Ls, vF2 = valFoldLoss(output1,output2,output3,output4,output5,valLabel1,valLabel2,valLabel3,valLabel4,valLabel5,val2Ls)
				fLoss[1] = tmpF2
				makeValImg(output1[:12],'neut',subDir2[1],neutV2[:12])
				makeValImg(output2[:11],'baso',subDir2[1],basoV2[:11])
				makeValImg(output3[:8],'eosi',subDir2[1],eosiV2[:8])
				makeValImg(output4[:12],'lymfo',subDir2[1],lymphoV2[:12])
				makeValImg(output5[:10],'mono',subDir2[1],monoV2[:10])
				torch.save(model, modPath + "/fold2/" +str(batch*5)+"_"+str(tmpF2))
			output1,output2,output3,output4,output5,valLabel1,valLabel2,valLabel3,valLabel4,valLabel5 = next(output1,output2,output3,output4,output5,valLabel1,valLabel2,valLabel3,valLabel4,valLabel5)
			
			fold3LsL,tmpF3 = testFoldLoss(output1,output2,output3,output4,output5,valLabel1,valLabel2,valLabel3,valLabel4,valLabel5,fold3LsL)
			if tmpF3 < fLoss[2]:
				val3Ls, vF3 = valFoldLoss(output1,output2,output3,output4,output5,valLabel1,valLabel2,valLabel3,valLabel4,valLabel5,val3Ls)
				fLoss[2] = tmpF3
				makeValImg(output1[:12],'neut',subDir2[2],neutV3[:12])
				makeValImg(output2[:11],'baso',subDir2[2],basoV3[:11])
				makeValImg(output3[:8],'eosi',subDir2[2],eosiV3[:8])
				makeValImg(output4[:12],'lymfo',subDir2[2],lymphoV3[:12])
				makeValImg(output5[:10],'mono',subDir2[2],monoV3[:10])
				torch.save(model, modPath + "/fold3/" +str(batch*5)+"_"+str(tmpF3))
			output1,output2,output3,output4,output5,valLabel1,valLabel2,valLabel3,valLabel4,valLabel5 = next(output1,output2,output3,output4,output5,valLabel1,valLabel2,valLabel3,valLabel4,valLabel5)
			
			fold4LsL,tmpF4 = testFoldLoss(output1,output2,output3,output4,output5,valLabel1,valLabel2,valLabel3,valLabel4,valLabel5,fold4LsL)
			if tmpF4 < fLoss[3]:
				val4Ls, vF4 = valFoldLoss(output1,output2,output3,output4,output5,valLabel1,valLabel2,valLabel3,valLabel4,valLabel5,val4Ls)
				fLoss[3] = tmpF4
				makeValImg(output1[:12],'neut',subDir2[3],neutV4[:12])
				makeValImg(output2[:11],'baso',subDir2[3],basoV4[:11])
				makeValImg(output3[:8],'eosi',subDir2[3],eosiV4[:8])
				makeValImg(output4[:12],'lymfo',subDir2[3],lymphoV4[:12])
				makeValImg(output5[:10],'mono',subDir2[3],monoV4[:10])
				torch.save(model, modPath + "/fold4/" +str(batch*5)+"_"+str(tmpF4))
			output1,output2,output3,output4,output5,valLabel1,valLabel2,valLabel3,valLabel4,valLabel5 = next(output1,output2,output3,output4,output5,valLabel1,valLabel2,valLabel3,valLabel4,valLabel5)
			
			fold5LsL,tmpF5 = testFoldLoss(output1,output2,output3,output4,output5,valLabel1,valLabel2,valLabel3,valLabel4,valLabel5,fold5LsL)
			if tmpF5 < fLoss[4]:
				val5Ls, vF5 = valFoldLoss(output1,output2,output3,output4,output5,valLabel1,valLabel2,valLabel3,valLabel4,valLabel5,val5Ls)
				torch.save(model, modPath + "/fold5/" +str(batch*5)+"_"+str(tmpF5))
				makeValImg(output1[:12],'neut',subDir2[4],neutV5[:12])
				makeValImg(output2[:11],'baso',subDir2[4],basoV5[:11])
				makeValImg(output3[:8],'eosi',subDir2[4],eosiV5[:8])
				makeValImg(output4[:12],'lymfo',subDir2[4],lymphoV5[:12])
				makeValImg(output5[:10],'mono',subDir2[4],monoV5[:10])
				fLoss[4] = tmpF5
			del output1,output2,output3,output4,output5
			
			#Determine total validation loss and store in a list	
			print vF1+vF2+vF3+vF4+vF5
			valLsTot.append(vF1+vF2+vF3+vF4+vF5)	
			
			
	#Plot train and test loss over time
	plt.plot(batchIter,trainLossIter)
	plt.plot(batchIter2,testLossIter)
	plt.ylabel("Average loss")
	plt.xlabel("Amount of images seen (n)")
	plt.title("Training loss and test loss")
	plt.legend(('Training Loss', 'Test Loss'),loc='upper right')
	plt.savefig(modPath+'/loss.png', bbox_inches='tight')
	plt.clf()
	
	# Save lists with stored results.
	with open(modPath+"/"+str(epoch)+"Fold1ls.pickle",'wb') as fp:
		cp.dump(fold1LsL,fp)
	with open(modPath+"/"+str(epoch)+"fold2ls.pickle",'wb') as fp:
		cp.dump(fold2LsL,fp)
	with open(modPath+"/"+str(epoch)+"fold3ls.pickle",'wb') as fp:
		cp.dump(fold3LsL,fp)
	with open(modPath+"/"+str(epoch)+"fold4ls.pickle",'wb') as fp:
		cp.dump(fold4LsL,fp)
	with open(modPath+"/"+str(epoch)+"fold5ls.pickle",'wb') as fp:
		cp.dump(fold5LsL,fp)
	with open(modPath+"/"+str(epoch)+"val1LS.pickle",'wb') as fp:
		cp.dump(val1Ls,fp)
	with open(modPath+"/"+str(epoch)+"val2LS.pickle",'wb') as fp:
		cp.dump(val2Ls,fp)
	with open(modPath+"/"+str(epoch)+"val3LS.pickle",'wb') as fp:
		cp.dump(val3Ls,fp)
	with open(modPath+"/"+str(epoch)+"val4LS.pickle",'wb') as fp:
		cp.dump(val4Ls,fp)
	with open(modPath+"/"+str(epoch)+"val5LS.pickle",'wb') as fp:
		cp.dump(val5Ls,fp)
	with open(modPath+"/"+str(epoch)+"valLSTot.pickle",'wb') as fp:
		cp.dump(valLsTot,fp)



if __name__ == "__main__":
	# Create directories to store results in.
	modPath = '../segmentation_results/model2/run9'
	subDir = ['/fold1','/fold2','/fold3','/fold4','/fold5']
	subDir2 = ['/fold1val','/fold2val','/fold3val','/fold4val','/fold5val']
	try:
		os.makedirs(modPath)
		for i in subDir:
			os.makedirs(modPath+i)
		for i in subDir2:
			os.makedirs(modPath+i)
	except OSError: pass
	# Use GPU if available otherwise CPU
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = ResBlock().to(device)
	criterion = torch.nn.BCELoss()
	optimizer = torch.optim.Adam(model.parameters())

	# Paths to synthetic training images and corresponding ground truth segmentation
	tPath1 = "../dataset/neut/cell/"
	tPath2 = "../dataset/baso/cell/"
	tPath3 = "../dataset/eosi/cell/"
	tPath4 = "../dataset/lympho/cell/"
	tPath5 = "../dataset/mono/cell/"

	lPath1 = "../dataset/neut/mask/"
	lPath2 = "../dataset/baso/mask/"
	lPath3 = "../dataset/eosi/mask/"
	lPath4 = "../dataset/lympho/mask/"
	lPath5 = "../dataset/mono/mask/"

	# Paths to each real cell image (LISC) and the corresponding ground truth segmentation.
	neutV = glob.glob('../validation_set2/neut120/*')
	basoV = glob.glob('../validation_set2/baso120/*')
	eosiV = glob.glob('../validation_set2/eosi120/*')
	lymphoV = glob.glob('../validation_set2/lympho120/*')
	monoV = glob.glob('../validation_set2/mono120/*')
	neutVLab = glob.glob('../validation_set2/neutgts120/*')
	basoVLab = glob.glob('../validation_set2/basogts120/*')
	eosiVLab = glob.glob('../validation_set2/eosigts120/*')
	lymphoVLab = glob.glob('../validation_set2/lymphogts120/*')
	monoVLab = glob.glob('../validation_set2/monogts120/*')
	
	random.Random(652).shuffle(neutV)
	random.Random(342).shuffle(basoV)
	random.Random(142).shuffle(eosiV)
	random.Random(232).shuffle(lymphoV)
	random.Random(434).shuffle(monoV)
	random.Random(652).shuffle(neutVLab)
	random.Random(342).shuffle(basoVLab)
	random.Random(142).shuffle(eosiVLab)
	random.Random(232).shuffle(lymphoVLab)
	random.Random(434).shuffle(monoVLab)
	
	# Rearrange the list of paths so they are in correct order for the next fold. This is used to make the image with the real cell image and the segmentation output.
	neutV2,basoV2,eosiV2,lymphoV2,monoV2 = nextPath(neutV,basoV,eosiV,lymphoV,monoV)
	neutV3,basoV3,eosiV3,lymphoV3,monoV3 = nextPath(neutV2,basoV2,eosiV2,lymphoV2,monoV2)
	neutV4,basoV4,eosiV4,lymphoV4,monoV4 = nextPath(neutV3,basoV3,eosiV3,lymphoV3,monoV3)
	neutV5,basoV5,eosiV5,lymphoV5,monoV5 = nextPath(neutV4,basoV4,eosiV4,lymphoV4,monoV4)
	# List with initial loss. A high value is used so the first loss obtained over the stopping bins is always lower.
	fLoss = [100,100,100,100,100]
	# Dummy validation loss
	vF1 = 0
	vF2 = 0
	vF3 = 0
	vF4 = 0
	vF5 = 0
	#Actual loop that does all the work.
	epochs = 1
	for epoch in xrange(epochs):
		train(epoch,criterion,vF1,vF2,vF3,vF4,vF5)
