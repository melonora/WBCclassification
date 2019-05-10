from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from scipy import misc
import torch.nn as nn
import numpy as np
import torch
import os
import glob
import random
from funcs import *
import torchvision
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cPickle as cp

ri = random.randint

class Classify(nn.Module):

    def __init__(s):
        super(Classify, s).__init__()
        s.conv1 = nn.Conv2d(3,64, kernel_size=3)
        s.bn1 = nn.BatchNorm2d(64)
        s.conv2 = nn.Conv2d(64,128, kernel_size=3)
        s.bn2 = nn.BatchNorm2d(128)
        s.conv3 = nn.Conv2d(128,128, kernel_size=3)
        s.bn3 = nn.BatchNorm2d(128)
        s.fcl1 = nn.Linear(128*13*13,32)
        s.bn4 = nn.BatchNorm1d(32)
        s.fcl2 = nn.Linear(32,5)

    def forward(s,x):
        h1 = F.max_pool2d(F.relu(s.bn1(s.conv1(x))), 2, 2)   
        h2 = F.max_pool2d(F.relu(s.bn2(s.conv2(h1))), 2, 2)        
        h3 = F.max_pool2d(F.relu(s.bn3(s.conv3(h2))), 2, 2)       
        h3 = h3.view(-1,128*13*13)
        h4 = F.relu(s.bn4(s.fcl1(h3)))
        return s.fcl2(h4)

def train(epoch,criterion,score,fold1,fold2,fold3,fold4,fold5,valF1,valF2,valF3,valF4,valF5, batchSize = 60):
	#loading validation data (extracted from LISC database)
	valNeut = Variable(torch.cat(tuple(misc2torch(misc.imread(i)[:,:,:3]/255.).view(1,3,120,120) for i in neutV),0).float()).to(device)
	valBaso = Variable(torch.cat(tuple(misc2torch(misc.imread(i)[:,:,:3]/255.).view(1,3,120,120) for i in basoV),0).float()).to(device)
	valEosi = Variable(torch.cat(tuple(misc2torch(misc.imread(i)[:,:,:3]/255.).view(1,3,120,120) for i in eosiV),0).float()).to(device)
	valLympho = Variable(torch.cat(tuple(misc2torch(misc.imread(i)[:,:,:3]/255.).view(1,3,120,120) for i in lymphoV),0).float()).to(device)
	valMono = Variable(torch.cat(tuple(misc2torch(misc.imread(i)[:,:,:3]/255.).view(1,3,120,120) for i in monoV),0).float()).to(device)
	
	# Batchsize is 60. Per cell type (cBatchsize) the amount of synthetic training images is equal to 12.
	cBatchSize = batchSize/5
	train_loss = 0
	trainLossTemp = 0
	batchIter = []
	batchIter2 = []
	trainLossIter = []
	testLossIter = []
	fold1ls = []
	fold2ls = []
	fold3ls = []
	fold4ls = []
	fold5ls = []
	valLs = []
	
	# Creating list of indices for loading the synthetic training data in batches
	nindex = range(80000)
	random.shuffle(nindex)
	
	# Labels for synthetic training data. Neutrophils=0, basophils=1, eosinophils=2,lymphocytes=3 and monocytes=4
	labels = Variable(torch.zeros(batchSize)).long().to(device)
	labels[:12] = 0
	labels[12:24] = 1
	labels[24:36] = 2
	labels[36:48] = 3
	labels[48:] = 4
	
	# Same as above but for synthetic test data. Labels are the same
	tLabels = Variable(torch.zeros(500)).long().to(device)
	tLabels[:100] = 0
	tLabels[100:200] = 1
	tLabels[200:300] = 2
	tLabels[300:400] = 3
	tLabels[400:] = 4
	
	# Actual for loop for training the CNN.
	for batch in xrange(0,80000,cBatchSize):
		model.train()
		optimizer.zero_grad()
		
		#Sublist of indices is created. Synthetic training data is then loaded in.
		ls = [i for i in nindex[batch:batch+cBatchSize]]
		data1 = torch.cat(tuple(misc2torch(augment(misc.imread(tPath1+str(i)+".png")[:,:,:3]/255.)).view(1,3,120,120) for i in ls),0).float()
		data2 = torch.cat(tuple(misc2torch(augment(misc.imread(tPath2+str(i)+".png")[:,:,:3]/255.)).view(1,3,120,120) for i in ls),0).float()
		data3 = torch.cat(tuple(misc2torch(augment(misc.imread(tPath3+str(i)+".png")[:,:,:3]/255.)).view(1,3,120,120) for i in ls),0).float()
		data4 = torch.cat(tuple(misc2torch(augment(misc.imread(tPath4+str(i)+".png")[:,:,:3]/255.)).view(1,3,120,120) for i in ls),0).float()
		data5 = torch.cat(tuple(misc2torch(augment(misc.imread(tPath5+str(i)+".png")[:,:,:3]/255.)).view(1,3,120,120) for i in ls),0).float()
		data = Variable(torch.cat((data1,data2,data3,data4,data5),0)).to(device) 
		
		# Prevents error at the end of the epoch (400000 not being wholy divisible by batchsize of 60 resulting in an index error if not prevented).
		if data.size(0) != batchSize:
			continue
		#Feeding the training data to the network, calculating the loss, performing back propagation and updating the parameters.	
		output = model(data)
		loss = criterion(output,labels)
		loss.backward()
		optimizer.step()
		train_loss += loss.data
		
		# After every 300 training images. Test loss and validation scores are determined.
		if batch% 60 == 0 and batch != 0:
			# storing the average loss of the previous 5 training batches
			trainLossIter.append(((train_loss.cpu().numpy()-trainLossTemp)/5.))
			batchIter.append(batch*5)
			trainLossTemp = train_loss.cpu().numpy()
			
			model.eval()
			
			#Creating indices for loading synthetic test data, actual loading of synthetic test data, feeding test data to the CNN and determining and storing test loss.
			nindex2 = [ri(80000,99999) for i in xrange(100)]
			testNeut = Variable(torch.cat(tuple(misc2torch(misc.imread(tPath1+str(i)+".png")[:,:,:3]/255.).view(1,3,120,120) for i in nindex2),0).float()).to(device)
			testBaso = Variable(torch.cat(tuple(misc2torch(misc.imread(tPath2+str(i)+".png")[:,:,:3]/255.).view(1,3,120,120) for i in nindex2),0).float()).to(device)
			testEosi = Variable(torch.cat(tuple(misc2torch(misc.imread(tPath3+str(i)+".png")[:,:,:3]/255.).view(1,3,120,120) for i in nindex2),0).float()).to(device)
			testLympho = Variable(torch.cat(tuple(misc2torch(misc.imread(tPath4+str(i)+".png")[:,:,:3]/255.).view(1,3,120,120) for i in nindex2),0).float()).to(device)
			testMono = Variable(torch.cat(tuple(misc2torch(misc.imread(tPath5+str(i)+".png")[:,:,:3]/255.).view(1,3,120,120) for i in nindex2),0).float()).to(device)
			tData = Variable(torch.cat((testNeut,testBaso,testEosi,testLympho,testMono),0)).to(device)
			with torch.no_grad():
				output = model(tData)
				loss = criterion(output,tLabels)
				tLoss = loss.data.cpu().numpy()
				batchIter2.append(batch*5)
				testLossIter.append(tLoss)
			
			# Feeding real single cell images (LISC database) to the CNN. 
			with torch.no_grad():
				output1 = F.softmax(model(valNeut),dim = 1)
				output2 = F.softmax(model(valBaso),dim = 1)
				output3 = F.softmax(model(valEosi),dim = 1)
				output4 = F.softmax(model(valLympho),dim = 1)
				output5 = F.softmax(model(valMono),dim = 1)
			
			# Indices correspond to the index of the maximum probability (values), this can be 0,1,2,3 or 4 and corresponds to the cell type label. 
			values1, indices1 = output1.max(1)
			values2, indices2 = output2.max(1)
			values3, indices3 = output3.max(1)
			values4, indices4 = output4.max(1)
			values5, indices5 = output5.max(1)
			
			# foldtmp corresponds to the f1 scores over the stopping bins per fold. The function nextFold is used to rearrange the list of indices (results) in order to obtain the results of the next fold.
			fold1tmp = testFold(indices1,indices2,indices3,indices4,indices5)
			ind12, ind22, ind32, ind42, ind52 = nextFold(indices1,indices2,indices3,indices4,indices5)
			fold2tmp = testFold(ind12, ind22, ind32, ind42, ind52)
			ind13,ind23,ind33,ind43,ind53 = nextFold(ind12, ind22, ind32, ind42, ind52)
			fold3tmp = testFold(ind13,ind23,ind33,ind43,ind53)
			ind14,ind24,ind34,ind44,ind54 = nextFold(ind13,ind23,ind33,ind43,ind53)
			fold4tmp = testFold(ind14,ind24,ind34,ind44,ind54)
			ind15,ind25,ind35,ind45,ind55 = nextFold(ind14,ind24,ind34,ind44,ind54)
			fold5tmp = testFold(ind15,ind25,ind35,ind45,ind55)
			
			# Used to keep track of f1 scores over stopping bins of various folds over time.
			fold1ls.append(fold1tmp)
			fold2ls.append(fold2tmp)
			fold3ls.append(fold3tmp)
			fold4ls.append(fold4tmp)
			fold5ls.append(fold5tmp)	
			
			# Checking whether the current f1 scores over stopping bins are higher than the previous best f1 score over stopping bins of the specific fold. If so this score is stored and the validation f1 score of specific fold is determined, the results of the fold are stored in foldResults and the model is saved. The valF variables are dataframes indicating for each reference cell type, the amount of predicted cell types.
			if fold1tmp > fold1:
				tmpValF1, binF1 = valFold(indices1,indices2,indices3,indices4,indices5)
				foldResults[0] = binF1
				fold1 = fold1tmp
				valF1 = tmpValF1
				torch.save(model, modPath + "/fold1/" +str(epoch)+"_"+str(batch*5)+"_"+str(fold1tmp))
			if fold2tmp > fold2:
				tmpValF2, binF2 = valFold(ind12, ind22, ind32, ind42, ind52)
				foldResults[1] = binF2
				fold2 = fold2tmp
				valF2 = tmpValF2
				torch.save(model, modPath + "/fold2/" +str(epoch)+"_" +str(batch*5)+"_"+str(fold2tmp))
			if fold3tmp > fold3:
				tmpValF3, binF3 = valFold(ind13,ind23,ind33,ind43,ind53)
				foldResults[2] = binF3
				fold3 = fold3tmp
				valF3 = tmpValF3
				torch.save(model, modPath + "/fold3/"+str(epoch)+"_" +str(batch*5)+"_"+str(fold3tmp))
			if fold4tmp > fold4:
				tmpValF4, binF4 = valFold(ind14,ind24,ind34,ind44,ind54)
				foldResults[3] = binF4
				fold4 = fold4tmp
				valF4 = tmpValF4
				torch.save(model, modPath + "/fold4/"+str(epoch)+"_" +str(batch*5)+"_"+str(fold4tmp))
			if fold5tmp > fold5:
				tmpValF5, binF5 = valFold(ind15,ind25,ind35,ind45,ind55)
				foldResults[4] = binF5
				fold5 = fold5tmp
				valF5 = tmpValF5
				torch.save(model, modPath + "/fold5/"+str(epoch)+"_" +str(batch*5)+"_"+str(fold5tmp))
			
			# Concatenating the results of the validation bins of all folds in order to determine a joint f1 score, which is stored in a list.	
			allFold = valF1+valF2+valF3+valF4+valF5
			f1 = f1Score(allFold)
			valLs.append(f1)
			
			# Printing output to the terminal to be able to check that the model is progressing with training.
			if batch%1680 == 0 and batch !=0:
				print epoch,batch
				print f1
	
	# training and test loss plot			
	plt.plot(batchIter,trainLossIter)
	plt.plot(batchIter2,testLossIter)
	plt.ylabel("Average loss")
	plt.xlabel("Amount of images seen (n)")
	plt.title("Training loss and test loss")
	plt.legend(('Training Loss', 'Test Loss'),loc='upper right')
	plt.savefig(modPath+"/"+str(epoch)+"loss.png", bbox_inches='tight')
	plt.clf()
	
	#Heatmaps for the validation bin of each fold and all folds together to indicate what cell types were predicted per reference cell type.
	allFold = valF1+valF2+valF3+valF4+valF5
	f1 = f1Score(allFold)
	heatmap(allFold,modPath+"/"+str(epoch)+"overallValF1_"+str(f1)+".png")
	heatmap(valF1,modPath+"/fold1_"+str(fold1)+".png")
	heatmap(valF2,modPath+"/fold2_"+str(fold2)+".png")
	heatmap(valF3,modPath+"/fold3_"+str(fold3)+".png")
	heatmap(valF4,modPath+"/fold4_"+str(fold4)+".png")
	heatmap(valF5,modPath+"/fold5_"+str(fold5)+".png")
	
	# plot of fold f1 scores of stopping bins over time as well as joint f1 score of validation bins.
	plt.plot(batchIter2,fold1ls)
	plt.plot(batchIter2,fold2ls)
	plt.plot(batchIter2,fold3ls)
	plt.plot(batchIter2,fold4ls)
	plt.plot(batchIter2,fold5ls)
	plt.plot(batchIter2,valLs)
	plt.ylabel("F1 score")
	plt.xlabel("Amount of images seen (n)")
	plt.title("Test and validation F1 scores of the folds")
	plt.legend(('Fold1', 'Fold2','Fold3','Fold4','Fold5','Validation'),loc='upper right')
	plt.savefig(modPath+"/"+str(epoch)+"f1scores.png", bbox_inches='tight')
	plt.clf()
	
	print foldResults
	# Storing lists containing results over time.
	with open(modPath+"/valBin_"+str(epoch)+"_"+str(f1)+".pickle",'wb') as fp:
		cp.dump(foldResults,fp)
	with open(modPath+"/"+str(epoch)+"Fold1ls.pickle",'wb') as fp:
		cp.dump(fold1ls,fp)
	with open(modPath+"/"+str(epoch)+"fold2ls.pickle",'wb') as fp:
		cp.dump(fold2ls,fp)
	with open(modPath+"/"+str(epoch)+"fold3ls.pickle",'wb') as fp:
		cp.dump(fold3ls,fp)
	with open(modPath+"/"+str(epoch)+"fold4ls.pickle",'wb') as fp:
		cp.dump(fold4ls,fp)
	with open(modPath+"/"+str(epoch)+"fold5ls.pickle",'wb') as fp:
		cp.dump(fold5ls,fp)
	with open(modPath+"/"+str(epoch)+"valLS.pickle",'wb') as fp:
		cp.dump(valLs,fp)
		
	return fold1,fold2,fold3,fold4,fold5,valF1,valF2,valF3,valF4,valF5

def heatmap(dataframe,savePath):
	# Creating the heatmap indicating per reference cell type the amount certain predictions were given by the CNN model.
	fig = sns.heatmap(dataframe,annot=True, annot_kws={"size": 7})
	plt.xlabel("Predicted cell type")
	plt.ylabel("Reference cell type")
	plt.title("Confusion matrix of cell type classification")
	fig.figure.savefig(savePath)
	plt.clf()
	
def count(indices,classN,y):
	# Counting the indices of max probabilities given by the CNN model. For example if 3 is encountered 3 times in indices, eosinophils are predicted 3 times. binLS indicates per image whether it was correctly classified (1) or not (0).
	lsCount1 = [0 for i in xrange(classN)] 
	indicesCPU = indices.cpu().data
	binLs = [0 for i in xrange(len(indicesCPU))]
	for i in xrange(len(indicesCPU)):
		lsCount1[indicesCPU[i]] += 1
		if y == indicesCPU[i]:
			binLs[i] = 1
	return lsCount1, binLs

def valFold(ls1,ls2,ls3,ls4,ls5,fold=5):
	#Slice index used to get results for the validation bins per fold
	intLs1 = int(math.ceil(len(ls1)/float(fold)))
	intLs2 = int(math.ceil(len(ls2)/float(fold)))
	intLs3 = int(math.ceil(len(ls3)/float(fold)))
	intLs4 = int(math.ceil(len(ls4)/float(fold)))
	intLs5 = int(math.ceil(len(ls5)/float(fold)))
	
	#Actual validation results per fold.	
	nF1TAbs, nF1Tbin = count(ls1[:intLs1],5,0)
	bF1TAbs, bF1Tbin = count(ls2[:intLs2],5,1)
	eF1TAbs, eF1Tbin = count(ls3[:intLs3],5,2)
	lF1TAbs, lF1Tbin = count(ls4[:intLs4],5,3)
	mF1TAbs, mF1Tbin = count(ls5[:intLs5],5,4)
	
	# DataFrame with the amount of predictions per cell type per reference cell type.
	dfF1 = pd.DataFrame(np.array([nF1TAbs,bF1TAbs,eF1TAbs,lF1TAbs,mF1TAbs]),index=["neut","baso","eosi","lymph","mono"], columns=["neut","baso","eosi","lymph","mono"])
	return dfF1, [nF1Tbin,bF1Tbin,eF1Tbin,lF1Tbin,mF1Tbin]
	
def nextFold(ls1,ls2,ls3,ls4,ls5,fold=5):
	#Slice index used to rearrange results to obtain the next fold
	intLs1 = int(math.ceil(len(ls1)/float(fold)))
	intLs2 = int(math.ceil(len(ls2)/float(fold)))
	intLs3 = int(math.ceil(len(ls3)/float(fold)))
	intLs4 = int(math.ceil(len(ls4)/float(fold)))
	intLs5 = int(math.ceil(len(ls5)/float(fold)))
	
	#Rearranging the indices to obtain the results of the next fold.
	ls12 = torch.cat((ls1[intLs1:],ls1[:intLs1]),0)
	ls22 = torch.cat((ls2[intLs2:],ls2[:intLs2]),0)
	ls32 = torch.cat((ls3[intLs3:],ls3[:intLs3]),0)
	ls42 = torch.cat((ls4[intLs4:],ls4[:intLs4]),0)
	ls52 = torch.cat((ls5[intLs5:],ls5[:intLs5]),0)
	return ls12,ls22,ls32,ls42,ls52
		
def testFold(ls1,ls2,ls3,ls4,ls5,fold=5):
	#Slice index used to get results for the stopping bins per fold
	intLs1 = int(math.ceil(len(ls1)/float(fold)))
	intLs2 = int(math.ceil(len(ls2)/float(fold)))
	intLs3 = int(math.ceil(len(ls3)/float(fold)))
	intLs4 = int(math.ceil(len(ls4)/float(fold)))
	intLs5 = int(math.ceil(len(ls5)/float(fold)))
	
	#Actual results of stopping bins per fold.
	nF1TAbs, nF1Tbin = count(ls1[intLs1:],5,0)
	bF1TAbs, bF1Tbin = count(ls2[intLs2:],5,1)
	eF1TAbs, eF1Tbin = count(ls3[intLs3:],5,2)
	lF1TAbs, lF1Tbin = count(ls4[intLs4:],5,3)
	mF1TAbs, mF1Tbin = count(ls5[intLs5:],5,4)
	
	# DataFrame with the amount of predictions per cell type per reference cell type. This is used to calculate the f1 score over the stopping bins of each fold.
	dfF1 = pd.DataFrame(np.array([nF1TAbs,bF1TAbs,eF1TAbs,lF1TAbs,mF1TAbs]),index=["neut","baso","eosi","lymph","mono"], columns=["neut","baso","eosi","lymph","mono"])
	foldScore = f1Score(dfF1)
	return foldScore
		
def f1Score(dataframe):
	#Calculate precision and recall in order to determine f1 score using the dataframe with the amount of cell type predictions per reference cell type.
	precision = 0
	recall = 0

	for i in xrange(len(dataframe)):
		if dataframe.iloc[:,i].sum() != 0: 
			subPrecision = dataframe.iloc[i,i]/float(dataframe.iloc[:,i].sum())
			precision += subPrecision
		else: pass
		subRecall = dataframe.iloc[i,i]/float(dataframe.iloc[i,:].sum())
		recall += subRecall
		
	precision = precision/float(len(dataframe))
	recall = recall/float(len(dataframe))
	f1 = 2*precision*recall/(recall+precision)
	return f1
	 
if __name__ == "__main__":
	#Creating directory to store results in
	modPath = '../classification_results/model3/runx'
	subDir = ['/fold1','/fold2','/fold3','/fold4','/fold5']
	try:
		os.makedirs(modPath)
		for i in subDir:
			os.makedirs(modPath+i)
	except OSError: pass
	
	# using GPU if available and otherwise CPU	
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = Classify().to(device)
	
	#
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters())
	
	#Paths of directories containing the synthetic image data
	tPath1 = "../dataset/neut/cell/"
	tPath2 = "../dataset/baso/cell/"
	tPath3 = "../dataset/eosi/cell/"
	tPath4 = "../dataset/lympho/cell/"
	tPath5 = "../dataset/mono/cell/"
	
	#Paths of all single cell images extracted from LISC database.
	neutV = glob.glob('../validation_set2/neut120/*')
	basoV = glob.glob('../validation_set2/baso120/*')
	eosiV = glob.glob('../validation_set2/eosi120/*')
	lymphoV = glob.glob('../validation_set2/lympho120/*')
	monoV = glob.glob('../validation_set2/mono120/*')
	
	# shuffling with seed the paths of single cell images. Same seed is used for each run.
	random.Random(652).shuffle(neutV)
	random.Random(342).shuffle(basoV)
	random.Random(142).shuffle(eosiV)
	random.Random(232).shuffle(lymphoV)
	random.Random(434).shuffle(monoV)
	
	#dummy list used to store the results of each validation image.
	subFoldResults = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
	foldResults = [subFoldResults for i in xrange(5)]
	
	# Dummy dataframes used to store the amount of predictions per cell type for each reference cell type. Fold scores are used to compare the validation F1 score obtained at a certain moment to the previous best validation F1 fold score.
	valF1 = pd.DataFrame(np.array([[0 for i in xrange(5)] for s in xrange(5)]),index=["neut","baso","eosi","lymph","mono"], columns=["neut","baso","eosi","lymph","mono"])
	valF2 = pd.DataFrame(np.array([[0 for i in xrange(5)] for s in xrange(5)]),index=["neut","baso","eosi","lymph","mono"], columns=["neut","baso","eosi","lymph","mono"])
	valF3 = pd.DataFrame(np.array([[0 for i in xrange(5)] for s in xrange(5)]),index=["neut","baso","eosi","lymph","mono"], columns=["neut","baso","eosi","lymph","mono"])
	valF4 = pd.DataFrame(np.array([[0 for i in xrange(5)] for s in xrange(5)]),index=["neut","baso","eosi","lymph","mono"], columns=["neut","baso","eosi","lymph","mono"])
	valF5 = pd.DataFrame(np.array([[0 for i in xrange(5)] for s in xrange(5)]),index=["neut","baso","eosi","lymph","mono"], columns=["neut","baso","eosi","lymph","mono"])
	fold1 = 0
	fold2 = 0
	fold3 = 0
	fold4 = 0
	fold5 = 0
	
	score = 0
	epochs = 3
	for epoch in xrange(epochs):
		fold1,fold2,fold3,fold4,fold5,valF1,valF2,valF3,valF4,valF5 = train(epoch,criterion,score,fold1,fold2,fold3,fold4,fold5,valF1,valF2,valF3,valF4,valF5)
