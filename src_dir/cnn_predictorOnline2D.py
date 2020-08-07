#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np

import functools

import time

import torch
from torch.autograd import Variable
from torch.nn       import Linear, ReLU, CrossEntropyLoss, \
                           Sequential, Conv2d, MaxPool2d,  \
                           Module, Softmax, BatchNorm2d, Dropout

from torch.optim    import Adam, SGD

from src_dir.cnn_collectionOnline2D import CnnOnline_2D

from src_dir import resid,timer,moving_average,GMRES





class CNNPredictorOnline_2D(object):

    def __init__(self,D_in,H,D_out):
        
        # N is batch size; D_in is input dimension;
        # H is hidden dimension; H2 is  second hidden layer dimension. 
        # D_out is output dimension.
        self.D_in=D_in
        self.H=H   
        self.D_out=D_out


        # Construct our model by instantiating the class defined above
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = CnnOnline_2D(self.D_in, self.H,self.D_out).to(device)

        # Construct our loss function and an Optimizer. The call to model.parameters()
        # in the SGD constructor will contain the learnable parameters of the two
        # nn.Conv1d modules which are members of the model.
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=1e-2)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)



        # x will hold entire training set b data
        # y will hold entire training set solution data
        self.x = torch.empty(0, self.D_in,self.D_in).to(device)
        self.y = torch.empty(0, self.D_out,self.D_out).to(device)



        # xNew holds new b additions to training set at the current time
        # yNew holds new solution (x) additions to training set at the current time
        self.xNew = torch.empty(0, self.D_in,self.D_in)
        self.yNew = torch.empty(0, self.D_out,self.D_out)


        self._is_trained = False

        # Diagnostic data => remove in production
        self.loss_val = list()

    @property
    def is_trained(self):
        return self._is_trained


    @is_trained.setter
    def is_trained(self, value):
        self._is_trained = value


    @property
    def counter(self):
        # Counter is based off of the data set to be added to training set 
        return self.xNew.size(0)


    @timer
    def retrain_timed(self):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.xNew = self.xNew.to(device)
        self.yNew = self.yNew.to(device)

        self.x=torch.cat((self.x,self.xNew))
        self.y=torch.cat((self.y,self.yNew))

        self.loss_val = list()  # clear loss val history
        self.loss_val.append(10.0)

        batch_size=32
        numEpochs=1000
        e1=1e-5
        epoch=0
        
        while self.loss_val[-1]> e1 and epoch<numEpochs-1:
            permutation = torch.randperm(self.x.size()[0])
            for t in range(0,self.x.size()[0],batch_size):
                
                indices = permutation[t:t+batch_size]

                batch_x, batch_y = self.x[indices],self.y[indices]

                # Forward pass: Compute predicted y by passing x to the model
                y_pred = self.model(batch_x)

                # Compute and print loss
                loss = self.criterion(y_pred, batch_y)
                self.loss_val.append(loss.item())

                # Zero gradients, perform a backward pass, and update the weights.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch=epoch+1

        permutation = torch.randperm(self.x.size()[0])
        indices = permutation[0:0+batch_size]
        batch_x, batch_y = self.x[indices],self.y[indices]
        # Adding new data to each batch
        # Note: only adding at most 3 data points to each batch
        batch_xMix=torch.cat((batch_x,self.xNew)) 
        batch_yMix=torch.cat((batch_y,self.yNew))
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = self.model(batch_xMix)
        # Compute and print loss
        loss = self.criterion(y_pred, batch_yMix)
        self.loss_val.append(loss.item())
        # Zero gradients, perform a backward pass, and update the weights.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
                
        print('Final loss:',loss.item())
        self.loss_val.append(loss.item())
        self.xNew = torch.empty(0, self.D_in,self.D_in)
        self.yNew = torch.empty(0, self.D_out,self.D_out)

        numparams=sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('parameters',numparams)

        self.is_trained = True




    def add(self, x, y):
        # TODO: don't use `torch.cat` in this incremental mode => will scale poorly
        # instead: use batched buffers
        self.xNew = torch.cat((self.xNew, torch.from_numpy(x).unsqueeze_(0).float()), 0)
        self.yNew = torch.cat((self.yNew, torch.from_numpy(y).unsqueeze_(0).float()), 0)

    def add_init(self, x, y):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.x = torch.cat((self.x, torch.from_numpy(x).unsqueeze_(0).float().to(device)), 0)
        self.y = torch.cat((self.y, torch.from_numpy(y).unsqueeze_(0).float().to(device)), 0)


    def predict(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        a1=torch.from_numpy(x).unsqueeze_(0).float().to(device)
        a2=np.squeeze(self.model.forward(a1).detach().cpu().numpy()) 
        #a2=np.squeeze(self.model.forward(a1).detach().numpy())     # cpu version, above line may work for cpu only... not sure. 
        return a2
        # inputs need to be [[x_1, x_2, ...]] as floats
        # outputs need to be numpy (non-grad => detach)
        # outputs need to be [y_1, y_2, ...]



def cnn_preconditionerOnline_timed_2D(retrain_freq=10, debug=False,InputDim=2,HiddenDim=100,OutputDim=2 ):
    def my_decorator(func):
        func.predictor    = CNNPredictorOnline_2D(InputDim,HiddenDim,OutputDim)
        func.retrain_freq = retrain_freq
        func.debug        = debug
        func.InputDim     = InputDim
        func.HiddenDim    = HiddenDim
        func.OutputDim    = OutputDim

        @functools.wraps(func)
        def speedup_wrapper(*args, **kwargs):

            A, b, x0, e, nmax_iter,ML_GMRES_Time_list,ProbCount,restart,debug,refine,blist,reslist,Err_list,reslist_flat,ML_GMRES_Time_list2,solIndx, *eargs = args

            trainTime=0.0
            IterTime=0
            
            
            Initial_set=2
            SpeedCutOff=0.00 #only add data is solution takes longer than 0.11 for coarse run

            IterTime_AVG=0.0
            IterErr10_AVG=0.0
            
            # Check if we are in first GMRES e1 tolerance run. If so, we compute prediction, and check the prediction is "good" before moving forward. 
            if func.predictor.is_trained and refine==False:
                pred_x0 = func.predictor.predict(b)
                # target_test  = func(A, b, pred_x0, e, nmax_iter,ML_GMRES_Time_list,ProbCount,1,debug,refine,blist,reslist,Err_list,reslist_flat, *eargs)
                target_test=GMRES(A, b, x0, e, 6,1, True)
                IterErr_test = resid(A, target_test, b)
                print('size',len(IterErr_test))
                print(IterErr_test[5],max(Err_list))
                if (IterErr_test[5]>max(Err_list)): 
                    print('poor prediction,using initial x0')
                    pred_x0 = x0
            else:
                pred_x0 = x0


            #Time GMRES function 
            tic = time.perf_counter()
            target  = func(A, b, pred_x0, e, nmax_iter,ML_GMRES_Time_list,ProbCount,restart,debug,refine,blist,reslist,Err_list,reslist_flat,ML_GMRES_Time_list2,solIndx, *eargs)
            toc = time.perf_counter()

            res = target[-1]


            # Check if we are in first e tolerance loop
            if refine==False :
                IterErr = resid(A, target, b)
                IterTime=(toc-tic)
                IterErr10=IterErr[5]
                ML_GMRES_Time_list.append(IterTime)
                Err_list.append(IterErr10)  

                if ProbCount<=Initial_set:
                    func.predictor.add_init(b, res)
                if ProbCount==Initial_set:
                    timeLoop=func.predictor.retrain_timed()
                    print('Initial Training')
            else :
                ML_GMRES_Time_list2.append((toc-tic))

            # Compute moving averages used to filter data
            if ProbCount>Initial_set:
                IterTime_AVG=moving_average(np.asarray(ML_GMRES_Time_list),ProbCount)
                IterErr10_AVG=moving_average(np.asarray(Err_list),ProbCount)
                print(ML_GMRES_Time_list[-1],IterTime_AVG,Err_list[-1],IterErr10_AVG)


            # Filter for data to be added to training set
            if (ML_GMRES_Time_list[-1]>IterTime_AVG and Err_list[-1]>IterErr10_AVG  ) and  refine==True and ProbCount>Initial_set and ML_GMRES_Time_list[-1]>SpeedCutOff  : 
                
                blist.append(b)
                reslist.append(res)
                reslist_flat.append(np.reshape(res,(1,-1),order='C').squeeze(0))

                fnameb='DataOuput/bdata/b'+str(solIndx).zfill(4)+'.dat'
                fnamesol='DataOuput/soldata/res'+str(solIndx).zfill(4)+'.dat'
                np.savetxt(fnameb,b)
                np.savetxt(fnamesol,res)
                
                # check orthogonality of 3 solutions that met training set critera
                if   len(blist)==3 :
                    resMat=np.asarray(reslist_flat)
                    resMat_square=resMat**2
                    row_sums = resMat_square.sum(axis=1,keepdims=True)
                    resMat= resMat/np.sqrt(row_sums)
                    InnerProd=np.dot(resMat,resMat.T)
                    print('InnerProd',InnerProd)

                    func.predictor.add(np.asarray(blist)[0], np.asarray(reslist)[0])
                    solIndx=solIndx+1
                    fnameb='DataOuput/bdata/b'+str(solIndx).zfill(4)+'.dat'
                    fnamesol='DataOuput/soldata/res'+str(solIndx).zfill(4)+'.dat'
                    np.savetxt(fnameb,blist[0])
                    np.savetxt(fnamesol,reslist[0])


                    cutoff=0.8
                    # Picking out sufficiently orthogonal subset of 3 solutions gathered
                    if np.abs(InnerProd[0,1]) and np.abs(InnerProd[0,2])<cutoff :
                        if np.abs(InnerProd[1,2])<cutoff :

                            func.predictor.add(np.asarray(blist)[1], np.asarray(reslist)[1])
                            solIndx=solIndx+1
                            fnameb='DataOuput/bdata/b'+str(solIndx).zfill(4)+'.dat'
                            fnamesol='DataOuput/soldata/res'+str(solIndx).zfill(4)+'.dat'
                            np.savetxt(fnameb,blist[1])
                            np.savetxt(fnamesol,reslist[1])

                            func.predictor.add(np.asarray(blist)[2], np.asarray(reslist)[2])
                            solIndx=solIndx+1
                            fnameb='DataOuput/bdata/b'+str(solIndx).zfill(4)+'.dat'
                            fnamesol='DataOuput/soldata/res'+str(solIndx).zfill(4)+'.dat'
                            np.savetxt(fnameb,blist[2])
                            np.savetxt(fnamesol,reslist[2])

                        elif np.abs(InnerProd[1,2])>=cutoff: 
                            func.predictor.add(np.asarray(blist)[1], np.asarray(reslist)[1])
                            solIndx=solIndx+1
                            fnameb='DataOuput/bdata/b'+str(solIndx).zfill(4)+'.dat'
                            fnamesol='DataOuput/soldata/res'+str(solIndx).zfill(4)+'.dat'
                            np.savetxt(fnameb,blist[1])
                            np.savetxt(fnamesol,reslist[1])

                    elif np.abs(InnerProd[0,1])<cutoff :
                        func.predictor.add(np.asarray(blist)[1], np.asarray(reslist)[1])
                        solIndx=solIndx+1
                        fnameb='DataOuput/bdata/b'+str(solIndx).zfill(4)+'.dat'
                        fnamesol='DataOuput/soldata/res'+str(solIndx).zfill(4)+'.dat'
                        np.savetxt(fnameb,blist[1])
                        np.savetxt(fnamesol,reslist[1])

                    elif np.abs(InnerProd[0,2])<cutoff :
                        func.predictor.add(np.asarray(blist)[2], np.asarray(reslist)[2])
                        solIndx=solIndx+1
                        fnameb='DataOuput/bdata/b'+str(solIndx).zfill(4)+'.dat'
                        fnamesol='DataOuput/soldata/res'+str(solIndx).zfill(4)+'.dat'
                        np.savetxt(fnameb,blist[2])
                        np.savetxt(fnamesol,reslist[2])


                # #  check orthogonality of 2 solutions that met training set critera
                # if   len(blist)==2 :
                #     resMat=np.asarray(reslist_flat)
                #     resMat_square=resMat**2
                #     row_sums = resMat_square.sum(axis=1,keepdims=True)
                #     resMat= resMat/np.sqrt(row_sums)
                #     InnerProd=np.dot(resMat,resMat.T)
                #     print('InnerProd',InnerProd)
                #     func.predictor.add(np.asarray(blist)[0], np.asarray(reslist)[0])
                #     cutoff=0.8
                    
                #     # Picking out sufficiently orthogonal subset of 3 solutions gathered
                #     if np.abs(InnerProd[0,1]) <cutoff :
                #         func.predictor.add(np.asarray(blist)[1], np.asarray(reslist)[1])
                    

                    if func.predictor.counter>=retrain_freq:
                        if func.debug:
                            print("retraining")
                            print(func.predictor.counter)
                            timeLoop=func.predictor.retrain_timed()
                            trainTime=float(timeLoop[-1])
                            blist=[]
                            reslist=[]
                            reslist_flat=[]
            return target,ML_GMRES_Time_list,trainTime,blist,reslist,Err_list,reslist_flat,ML_GMRES_Time_list2,solIndx

        return speedup_wrapper
    return my_decorator