import tensorflow as tf
import numdifftools as nd
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
#np.version.version
from scipy.optimize import minimize
from random import shuffle
from sklearn.decomposition import PCA

df=pd.read_csv("WBC.csv")
data=np.array(df)
x_0 = data[0:239,0:-1]
x_1 = data[239:478,0:-1]

x_test_0 = x_0[180:239]
x_train_0 = x_0[0:180]

x_test_1 = x_1[180:239]
x_train_1 = x_1[0:180]
x_0.shape[1]

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

def f(param,x_0,x_1,test=False):
    num_dim=x_train_0.shape[1]
    w=param[0:num_dim]
    alpha=param[num_dim]
    beta=param[num_dim+1]
    gamma=param[num_dim+2]
    rho_class_A,rho_class_B=convert_to_density(x_0,x_1,w)
    result_class_A,result_class_B=apply_unitary_to_density_matrix(rho_class_A,rho_class_B,alpha,beta,gamma)
    p_class_A_of_result_class_A,p_class_B_of_result_class_B=measurement(result_class_A,result_class_B)
    #loss=-(1-p_class_A_of_result_class_A)*log(p_class_A_of_result_class_A)-(1-p_class_B_of_result_class_B)*log(p_class_B_of_result_class_B)
    
    loss=len(x_0)*(1-p_class_A_of_result_class_A)+len(x_1)*(1-p_class_B_of_result_class_B)
    if(test):
        return loss.real[0],p_class_A_of_result_class_A.real,p_class_B_of_result_class_B.real
    else:
        return loss.real[0]+np.sum(np.power(param,2))
    
def inferece():
    # training accuracy
    train_acc=0
    test_acc=0
    for i in tqdm.tqdm(range(len(x_train_0))):
        info=f(param,x_train_0[i],x_train_1[i],test=True)
        #print(info[1],info[2])
        if(info[1]>0.5):
            train_acc+=1
        if(info[2]>0.5):
            train_acc+=1
    for i in tqdm.tqdm(range(len(x_test_0))):
        info_t=f(param,x_test_0[i],x_test_1[i],test=True)
        if(info_t[1]>0.5):
            test_acc+=1
        if(info_t[2]>0.5):
            test_acc+=1
    train_acc=train_acc/(2*len(x_train_0))
    test_acc=test_acc/(2*len(x_test_0))
    return train_acc,test_acc

import tqdm
param=get_param(30,var=0.1)
losses=[]
losses_t=[]
accuracies_t=[]
accuracies=[]
grad=nd.Gradient(f)
m=np.random.random()
v=np.random.random()
for t in tqdm.tqdm(range(1,1000)):
    print(t)
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon= 1e-8
    g = grad(param,x_train_0[np.random.choice(len(x_train_0), 32)],x_train_1[np.random.choice(len(x_train_0), 32)])
    #print(g)
    m = beta_1 * m + (1 - beta_1) * g
    v = beta_2 * v + (1 - beta_2) * np.power(g, 2)
    m_hat = m / (1 - np.power(beta_1, t))
    v_hat = v / (1 - np.power(beta_2, t))
    param = param - (1e-2) * m_hat / (np.sqrt(v_hat) + epsilon)
    if(t%5==0):
        train_acc,test_acc=inferece()
        info=f(param,x_train_0,x_train_1,test=True)
        info_t=f(param,x_test_0,x_test_1,test=True)
        train_acc,test_acc=inferece()
        print("Loss : "+str(info[0][0]))
        print("Accuracy : "+str(train_acc)) # [0][0] since its returning a 2D array
        print("Test Loss : "+str(info_t[0][0]))
        print("Test Accuracy : "+str(test_acc)) # [0][0] since its returning a 2D array
        losses.append(info[0][0])
        accuracies.append(train_acc)
        losses_t.append(info_t[0][0])
        accuracies_t.append(test_acc)



losses=np.array(losses)
accuracies=np.array(accuracies)
losses_t=np.array(losses_t)
accuracies_t=np.array(accuracies_t)
data=[losses,losses_t,accuracies,accuracies_t,param] #784+3
print("--Training Finished--")
print("--Dumping necessary files--")
with open("wbc_adam.pickle",'wb') as pickle_file:
	pickle.dump(data,pickle_file,protocol=pickle.HIGHEST_PROTOCOL)

print("--Dumping Done--")


plt.plot(losses,label='Training accuracy')
plt.plot(losses_t,label='Testing accuracy')
plt.xlabel("Adam Iterations")
plt.ylabel("Training Loss")
plt.legend(loc='best')
plt.grid()
plt.suptitle("Learning Curve for wbc")
plt.savefig("Loss_wbc.png")
plt.show()