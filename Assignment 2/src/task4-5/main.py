import numpy as np
import matplotlib.pyplot as plt
import mnist
import tqdm

#THIS SHOULD REALLY HAVE BEEN DONE USING OOP

#The only reason I haven't implemented using classes
#is because implementing only 1 more hidden layer is
#less work than changing everything to oop. If more
#layers were needed this is not the way to do it

def should_early_stop(validation_loss, num_steps=3):
    if len(validation_loss) < num_steps+1:
        return False

    is_increasing = [validation_loss[i] <= validation_loss[i+1] for i in range(-num_steps-1, -1)]
    return sum(is_increasing) == len(is_increasing) 

def train_val_split(X, Y, val_percentage):
  """
    Selects samples from the dataset randomly to be in the validation set. Also, shuffles the train set.
    --
    X: [N, num_features] numpy vector,
    Y: [N, 1] numpy vector
    val_percentage: amount of data to put in validation set
  """
  dataset_size = X.shape[0]
  idx = np.arange(0, dataset_size)
  np.random.shuffle(idx) 
  
  train_size = int(dataset_size*(1-val_percentage))
  idx_train = idx[:train_size]
  idx_val = idx[train_size:]
  X_train, Y_train = X[idx_train], Y[idx_train]
  X_val, Y_val = X[idx_val], Y[idx_val]
  return X_train, Y_train, X_val, Y_val

def onehot_encode(Y, n_classes=10):
    onehot = np.zeros((Y.shape[0], n_classes))
    onehot[np.arange(0, Y.shape[0]), Y] = 1
    return onehot

def bias_trick(X):
    return np.concatenate((X, np.ones((len(X), 1))), axis=1)

def calc_loss(X, targets, w, layer_type):
    if(layer_type == 'hidden'):
      output = forward_to_hidden(X, w)
    elif(layer_type == 'out'):
      output = forward_to_out(X, w)

    assert output.shape == targets.shape 
    log_y = np.log(output)
    cross_entropy = -targets * log_y
    return cross_entropy.mean()

def check_gradient(X, targets, w, epsilon, computed_gradient, layer_type):
    print("Checking gradient...")
    dw = np.zeros_like(w)
    for k in range(w.shape[0]):
        for j in range(w.shape[1]):
            new_weight1, new_weight2 = np.copy(w), np.copy(w)
            new_weight1[k,j] += epsilon
            new_weight2[k,j] -= epsilon
            loss1 = calc_loss(X, targets, new_weight1, layer_type)
            loss2 = calc_loss(X, targets, new_weight2, layer_type)
            dw[k,j] = (loss1 - loss2) / (2*epsilon)
    maximum_abosulte_difference = abs(computed_gradient-dw).max()
    assert maximum_abosulte_difference <= epsilon**2, "Absolute error was: {}".format(maximum_abosulte_difference)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def der_sigmoid(a):
    return np.multiply(a, (1-a))

def improved_sigmoid(z):
  return 1.7159 * np.tanh((2/3)*z)

def der_improved_sigmoid(a):
  return 1.7159 * (2/3) * (1-np.tanh(2/3*a)**2)

def ReLU(z):
  return np.maximum(0,z)

def der_ReLU(a):
  a[a<=0] = 0
  a[a>0] = 1
  return a

def softmax(z):
    expz = np.exp(z)
    return expz / expz.sum(axis=1, keepdims=True)

def forward_to_hidden(X, w):
  if(should_improve_sigmoid):
    return improved_sigmoid(X.dot(w.T))
  elif(should_ReLU):
    return ReLU(X.dot(w.T))
  else:
    return sigmoid(X.dot(w.T))

def forward_to_out(H, w):
    return softmax(H.dot(w.T))

def calculate_accuracy(X, targets, w_ji, w_kj, w_lk):
    H1 = forward_to_hidden(X, w_ji)
    H2 = forward_to_hidden(H1, w_kj)
    output = forward_to_out(H2, w_lk)
    predictions = output.argmax(axis=1)
    targets = targets.argmax(axis=1)
    return (predictions == targets).mean()

def cross_entropy_loss(X, targets, w_ji, w_kj, w_lk):
    H1 = forward_to_hidden(X, w_ji)
    H2 = forward_to_hidden(H1, w_kj)
    output = forward_to_out(H2, w_lk)
    assert output.shape == targets.shape 
    log_y = np.log(output)
    cross_entropy = -targets * log_y
    return cross_entropy.mean()

def gradient_descent(H2, H1, X, targets, w_ji, w_kj, w_lk, learning_rate, should_check_gradient):
    normalization_factor = H1.shape[0] * targets.shape[1] # batch_size * num_classes
    outputs = forward_to_out(H2, w_lk)
    delta_k = - (targets - outputs)

    #Finding gradient between hidden layer H2 and output layer 

    dw_lk = np.dot(delta_k.T, H2)
    dw_lk = dw_lk / normalization_factor # Normalize gradient equally as loss normalization

    assert dw_lk.shape == w_lk.shape, "dw_lk shape was: {}. Expected: {}".format(dw_lk.shape, w_lk.shape)

    #Finding gradient between input H1 and hidden layer H2

    if(should_improve_sigmoid):
      f2_prime = der_improved_sigmoid(np.dot(H1, w_kj.T))
    elif(should_ReLU):
      f2_prime = der_ReLU(np.dot(H1, w_kj.T))
    else:
      f2_prime = der_sigmoid(np.dot(H1, w_kj.T))

    delta2_j = f2_prime*np.dot(delta_k, w_lk)

    dw_kj = np.dot(delta2_j.T, H1)
    dw_kj = dw_kj / normalization_factor # Normalize gradient equally as loss normalization

    assert dw_kj.shape == w_kj.shape, "dw_kj shape was: {}. Expected: {}".format(dw_kj.shape, w_kj.shape)

    #Finding gradient between input X and hidden layer H1

    if(should_improve_sigmoid):
      f1_prime = der_improved_sigmoid(np.dot(X, w_ji.T))
    elif(should_ReLU):
      f1_prime = der_ReLU(np.dot(X, w_ji.T))
    else:
      f1_prime = der_sigmoid(np.dot(X, w_ji.T))

    delta1_j = f1_prime*np.dot(delta2_j, w_kj)

    dw_ji = np.dot(delta1_j.T, X)
    dw_ji = dw_ji / normalization_factor # Normalize gradient equally as loss normalization

    assert dw_ji.shape == w_ji.shape, "dw_ji shape was: {}. Expected: {}".format(dw_ji.shape, w_ji.shape)

    

    if should_check_gradient:
        check_gradient(X, H1, w_ji, 10e-2,  dw_ji, 'hidden')
        check_gradient(H1, H2, w_kj, 10e-2,  dw_kj, 'hidden')
        check_gradient(H2, targets, w_lk, 10e-2,  dw_lk, 'out')

    #Updating weights
    w_ji = w_ji - learning_rate*dw_ji
    w_kj = w_kj - learning_rate*dw_kj
    w_lk = w_lk - learning_rate*dw_lk

    #Momentum
    if should_add_momentum:
      global w_ji_m, w_kj_m, w_lk_m
      w_ji = w_ji - momentum*w_ji_m
      w_kj = w_kj - momentum*w_kj_m  
      w_lk = w_lk - momentum*w_lk_m   
      w_ji_m = dw_ji
      w_kj_m = dw_kj
      w_lk_m = dw_lk

    return w_ji, w_kj, w_lk  

#Shuffling task 3
def shuffle(X, Y): 
    index = np.arange(0, X.shape[0])
    np.random.shuffle(index)

    return X[index], Y[index]

#mnist.init()
X_train, Y_train, X_test, Y_test = mnist.load()

# Pre-process data
X_train, X_test = X_train / 127.5 - 1, X_test / 127.5 - 1 #task 2b
X_train = bias_trick(X_train)
X_test = bias_trick(X_test)
Y_train, Y_test = onehot_encode(Y_train), onehot_encode(Y_test)

X_train, Y_train, X_val, Y_val = train_val_split(X_train, Y_train, 0.1)


# Hyperparameters
batch_size = 32
learning_rate = 0.5
num_batches = X_train.shape[0] // batch_size
check_step = num_batches // 10
max_epochs = 20
hidden_layer_size = 59

#For implementing momentum
momentum = 0.9
w_ji_m = 0
w_kj_m = 0
w_lk_m = 0

#Flags 
should_gradient_check = False
should_shuffle = True
should_normal_weights = True
should_add_momentum = True

#Activation function | ONLY ONE TRUE (or none)
should_improve_sigmoid = True
should_ReLU = False

# Tracking variables
TRAIN_LOSS = []
TEST_LOSS = []
VAL_LOSS = []
TRAIN_ACC = []
TEST_ACC = []
VAL_ACC = []

def train_loop():
    X_t, Y_t = X_train, Y_train

    #Initializing weights
    if should_normal_weights:
      w_ji = np.random.normal(0,1/np.sqrt(X_t.shape[1]),(hidden_layer_size, X_t.shape[1]))
      w_kj = np.random.normal(0,1/np.sqrt(hidden_layer_size),(hidden_layer_size, hidden_layer_size))
      w_lk = np.random.normal(0,1/np.sqrt(hidden_layer_size),(Y_t.shape[1], hidden_layer_size))
    else:
      w_ji = np.random.uniform(-1,1,(hidden_layer_size, X_t.shape[1]))
      w_kj = np.random.uniform(-1,1,(hidden_layer_size, hidden_layer_size))
      w_lk = np.random.uniform(-1,1,(Y_t.shape[1], hidden_layer_size))

    #Training
    for e in range(max_epochs): # Epochs
        for i in tqdm.trange(num_batches):
            X_batch = X_train[i*batch_size:(i+1)*batch_size]
            Y_batch = Y_train[i*batch_size:(i+1)*batch_size]
            H1_batch = forward_to_hidden(X_batch,w_ji)
            H2_batch = forward_to_hidden(H1_batch,w_kj)
            w_ji, w_kj, w_lk = gradient_descent(H2_batch, H1_batch, X_batch, Y_batch, w_ji, w_kj, w_lk, learning_rate, should_gradient_check)
            if i % check_step == 0:
                # Loss
                TRAIN_LOSS.append(cross_entropy_loss(X_train, Y_train, w_ji, w_kj, w_lk))
                TEST_LOSS.append(cross_entropy_loss(X_test, Y_test, w_ji, w_kj, w_lk))
                VAL_LOSS.append(cross_entropy_loss(X_val, Y_val, w_ji, w_kj, w_lk))
                
                # Accuracy
                TRAIN_ACC.append(calculate_accuracy(X_train, Y_train, w_ji, w_kj, w_lk))
                VAL_ACC.append(calculate_accuracy(X_val, Y_val, w_ji, w_kj, w_lk))
                TEST_ACC.append(calculate_accuracy(X_test, Y_test, w_ji, w_kj, w_lk))
                if should_early_stop(VAL_LOSS):
                    print(VAL_LOSS[-4:])
                    print("early stopping at epoch {}".format(e))
                    return w_ji, w_kj, w_lk
        print("Epoch: {} | Accuracy: {} | Loss :{}".format(e+1, round(VAL_ACC[-1],2), round(VAL_LOSS[-1],2)))
        X_t, Y_t = shuffle(X_train, Y_train)
    return w_ji, w_kj, w_lk


w_ji, w_kj, w_lk = train_loop()

plt.plot(TRAIN_LOSS, label="Training loss")
plt.plot(TEST_LOSS, label="Testing loss")
plt.plot(VAL_LOSS, label="Validation loss")
plt.legend()
plt.ylim()
plt.show()

plt.clf()
plt.plot(TRAIN_ACC, label="Training accuracy")
plt.plot(TEST_ACC, label="Testing accuracy")
plt.plot(VAL_ACC, label="Validation accuracy")
plt.ylim()
plt.legend()
plt.show()

plt.clf()








