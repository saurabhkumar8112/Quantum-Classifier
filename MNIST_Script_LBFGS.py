#necessary imports
import tensorflow as tf
import numdifftools as nd
import numpy as np
import pandas as pd
import tqdm
import pickle
import matplotlib.pyplot as plt
#np.version.version
from scipy.optimize import minimize
from random import shuffle
from sklearn.decomposition import PCA

(x_train, y_train), (x_test, y_test)=tf.keras.datasets.mnist.load_data(path='mnist.npz')

x_train=(x_train.reshape((x_train.shape[0],x_train.shape[1]*x_train.shape[2])))
x_test=(x_test.reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2])))

#pca = PCA(n_components=400)
#x_train=pca.fit_transform(x_train)
#x_test=pca.fit_transform(x_test)
#x_train.shape

#specify which class vs all classfication, for eg 0 vs all, oe 1 vs all
class_name=0
indx_0_train=[i for i, x in enumerate(y_train) if x == class_name]
#indx_3=np.random.choice(indx_3, size=5000, replace=False, p=None)
indx_1_train=[i for i, x in enumerate(y_train) if x != class_name]
#indx_5=np.random.choice(indx_5, size=1000, replace=False, p=None)
shuffle(indx_0_train)
shuffle(indx_1_train)

#taking equal number of samples to train
x_train_0 = (x_train[indx_0_train])/255.0
x_train_1 = (x_train[indx_1_train[0:len(indx_0_train)]])/255.0
print(x_train_0.shape,x_train_1.shape)

indx_0_test=[i for i, x in enumerate(y_test) if x == class_name]
#indx_3=np.random.choice(indx_3, size=5000, replace=False, p=None)
indx_1_test=[i for i, x in enumerate(y_test) if x != class_name]
#indx_5=np.random.choice(indx_5, size=1000, replace=False, p=None)
shuffle(indx_0_test)
shuffle(indx_1_test)

#taking equal number of samples to test
x_test_0 = x_test[indx_0_test]/255.0
x_test_1 = x_test[indx_1_test[0:len(indx_0_test)]]/255.0
#x_test_0=x_test_0[0:100]
#x_test_1=x_test_1[0:100]
print(x_test_0.shape,x_test_1.shape)

#function to get the initial
def get_param(num_param,var):
    tf.reset_default_graph()
    param=tf.get_variable(name="weight", dtype=tf.float32,shape=[num_param,],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=var))
    #param=tf.get_variable(name="weight", dtype=tf.float32,shape=[num_param,],initializer=tf.contrib.layers.xavier_initializer())
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    return param.eval(sess)


def to_single_number(w,x):
    #print(np.sum(x*w,axis=1).shape)
    mul=x*w
    #print(mul.shape)
    #print(np.sum(mul,axis=1).shape)
    try:
    	return np.sum(mul,axis=1) #use np.sum(mul) when inferencing
    except:
    	return np.sum(mul)


def convert_to_density(class_A,class_B,w):
    s_n_class_A=to_single_number(w,class_A)
    s_n_class_B=to_single_number(w,class_B)

    
    a_class_A = np.exp(np.complex(0,1)*np.array(s_n_class_A))
    a_conjugate_class_A = a_class_A.conj()
    
    a_square_class_A=a_class_A**2
    a_conj_square_class_A=a_conjugate_class_A**2
    square_sum_class_A=np.sum(a_square_class_A)/len(class_A)
    conj_sum_class_A=np.sum(a_conj_square_class_A)/len(class_A)
    
    rho_class_A=0.5*np.array([[1,square_sum_class_A], [conj_sum_class_A,1]])
    
    a_class_B = np.exp(np.complex(0,1)*s_n_class_B)
    a_conjugate_class_B = a_class_B.conj()
    
    a_square_class_B=a_class_B**2
    a_conj_square_class_B=a_conjugate_class_B**2
    square_sum_class_B=np.sum(a_square_class_B)/len(class_B)
    conj_sum_class_B=np.sum(a_conj_square_class_B)/len(class_B)
    
    rho_class_B=0.5*np.array([[1,square_sum_class_B], [conj_sum_class_B,1]])
    
    return rho_class_A, rho_class_B

def apply_unitary_to_density_matrix(rho_class_A,rho_class_B,alpha,beta,gamma):
    alpha_u = np.cos(alpha)*np.array([[1,0],[0,1]]) + np.complex(0.0,np.sin(alpha))*np.array([[1,0],[0,1]])
    beta_u = np.cos(beta)*np.array([[1,0],[0,1]]) + np.sin(beta)*np.array([[0,1],[-1,0]])
    gamma_u = np.cos(gamma)*np.array([[1,0],[0,1]]) + np.complex(0.0,np.sin(gamma))*np.array([[1,0],[0,1]])
    
    U = np.matmul(alpha_u,np.matmul(beta_u,gamma_u))
    U_dagger = np.transpose(np.conj(U))
    
    result_class_A = np.matmul(U,np.matmul(((rho_class_A)),U_dagger))
    result_class_A[0][0] = result_class_A[0][0].real
    result_class_A[1][1] = result_class_A[1][1].real
        
    result_class_B = np.matmul(U,np.matmul(rho_class_B,U_dagger))
    result_class_B[0][0] = result_class_B[0][0].real
    result_class_B[1][1] = result_class_B[1][1].real
    
    return result_class_A,result_class_B
        
def measurement(result_class_A,result_class_B):
    zero_ket = np.array([1,0]).reshape(2,1)
    one_ket = np.array([0,1]).reshape(2,1)
    
    p_class_A_of_result_class_A = np.matmul(zero_ket.T.conj(),np.matmul(result_class_A,zero_ket))
    p_class_B_of_result_class_B = np.matmul(one_ket.T.conj(),np.matmul(result_class_B,one_ket))
    
    return p_class_A_of_result_class_A.real,p_class_B_of_result_class_B.real

#th function used for minimization and optimization
def f(param,x_0,x_1,test=False):
    num_dim=x_train.shape[1]
    w=param[0:num_dim]
    alpha=param[num_dim]
    beta=param[num_dim+1]
    gamma=param[num_dim+2]
    rho_class_A,rho_class_B=convert_to_density(x_0,x_1,w)
    result_class_A,result_class_B=apply_unitary_to_density_matrix(rho_class_A,rho_class_B,alpha,beta,gamma)
    p_class_A_of_result_class_A,p_class_B_of_result_class_B=measurement(result_class_A,result_class_B)
    #loss=-(1-p_class_A_of_result_class_A)*log(p_class_A_of_result_class_A)-(1-p_class_B_of_result_class_B)*log(p_class_B_of_result_class_B)
    
    loss=len(x_0)*(1-p_class_A_of_result_class_A)+len(x_1)*(1-p_class_B_of_result_class_B)
    # if testing return accuracies along with the loss
    if(test):
        return loss.real[0],p_class_A_of_result_class_A.real,p_class_B_of_result_class_B.real
    else:
        return loss.real[0]


param=get_param(787,var=0.1) # uncomment/comment this for train/test 
losses=[]
losses_t=[]
accuracies_t=[]
accuracies=[]
def inferece(Xi):
	# training accuracy
	train_acc=0
	test_acc=0
	for i in tqdm.tqdm(range(len(x_train_0))):
		info=f(Xi,x_train_0[i],x_train_1[i],test=True)
		#print(info[1],info[2])
		if(info[1]>0.5):
			train_acc+=1
		if(info[2]>0.5):
			train_acc+=1
	for i in tqdm.tqdm(range(len(x_test_0))):
		info_t=f(Xi,x_test_0[i],x_test_1[i],test=True)
		if(info_t[1]>0.5):
			test_acc+=1
		if(info_t[2]>0.5):
			test_acc+=1
	train_acc=train_acc/(2*len(x_train_0))
	test_acc=test_acc/(2*len(x_test_0))
	return train_acc,test_acc

def callbackF(Xi):
    global losses
    global accuracies
    train_acc,test_acc=inferece(Xi)
    info=f(Xi,x_train_0,x_train_1,test=True)
    info_t=f(Xi,x_test_0,x_test_1,test=True)
    print("Loss : "+str(info[0][0]))
    print("Accuracy : "+str(train_acc)) # [0][0] since its returning a 2D array
    print("Loss : "+str(info_t[0][0]))
    print("Accuracy : "+str(test_acc)) # [0][0] since its returning a 2D array
    losses.append(info[0][0])
    accuracies.append(train_acc)
    losses_t.append(info_t[0][0])
    accuracies_t.append(test_acc)

# uncoment this block to start training
print("Starting training for "+str(class_name)+"vs all")
for i in tqdm.tqdm(range(10)):
	res=minimize(f,param,args=(x_train_0,x_train_1),callback=callbackF,method='L-BFGS-B')
	if(abs(f(param,x_train_0,x_train_1,test=True)[0][0]-losses[-1])<1):
		print("Condition True, Breaking")
		break
	param=res.x



losses=np.array(losses)
accuracies=np.array(accuracies)
losses_t=np.array(losses_t)
accuracies_t=np.array(accuracies_t)
data=[losses,losses_t,accuracies,accuracies_t,param]
print("--Training Finished--")
print("--Dumping necessary files--")
with open("data_"+str(class_name)+".pickle",'wb') as pickle_file:
	pickle.dump(data,pickle_file,protocol=pickle.HIGHEST_PROTOCOL)

print("--Dumping Done--")

#final test accuracy
info_t=f(param,x_test_0,x_test_1,test=True)
accuracy_t=info_t[1]*len(x_test_0)+info_t[2]*len(x_test_1)
accuracy_t/=(len(x_test_0)+len(x_test_1))
print("Test Loss : "+str(info_t[0][0]))
print("Test Accuracy : "+str(accuracy_t[0][0])) # [0][0] since its returning a 2D array

#now let's make a graph
plt.plot(accuracies,label='Training accuracy')
plt.plot(accuracies_t,label='Testing accuracy')
plt.xlabel("L-BFGS-B Iterations")
plt.ylabel("Training Loss")
plt.legend(loc='best')
plt.grid()
plt.suptitle("Learning Curve for "+str(class_name)+" vs all")
plt.savefig("Loss_"+str(class_name)+".png")
# print("Loading Data")
# with open("data_0.pickle",'rb') as pickle_file:
# 	data=pickle.load(pickle_file)
# param=data[-1]







    
