import numpy as np
import matplotlib.pyplot as plt
import mnist
import tqdm

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

def softmax(z):
  expz = np.exp(z)
  return expz / expz.sum(axis=1, keepdims=True)

def forward_to_hidden(X, w_ji):
  if(should_improve_sigmoid):
    return improved_sigmoid(X.dot(w_ji.T))
  else:
    return sigmoid(X.dot(w_ji.T))

def forward_to_out(H, w_kj):
    return softmax(H.dot(w_kj.T))

def calculate_accuracy(X, targets, w_ji, w_kj):
    H = forward_to_hidden(X, w_ji)
    output = forward_to_out(H, w_kj)
    predictions = output.argmax(axis=1)
    targets = targets.argmax(axis=1)
    return (predictions == targets).mean()

def cross_entropy_loss(X, targets, w_ji, w_kj):
    H = forward_to_hidden(X, w_ji)
    output = forward_to_out(H, w_kj)
    assert output.shape == targets.shape 
    log_y = np.log(output)
    cross_entropy = -targets * log_y
    return cross_entropy.mean()

def gradient_descent(H, X, targets, w_ji, w_kj, learning_rate, should_check_gradient):
    normalization_factor = H.shape[0] * targets.shape[1] # batch_size * num_classes
    outputs = forward_to_out(H, w_kj)
    delta_k = - (targets - outputs)

    #Finding gradient between input X and hidden layer H
    if(should_improve_sigmoid):
      f_prime = der_improved_sigmoid(H)
    else:
      f_prime = der_sigmoid(H)

    delta_j = np.multiply(f_prime, np.dot(delta_k, w_kj))
    dw_ji = np.dot(delta_j.T, X)
    dw_ji = dw_ji / normalization_factor # Normalize gradient equally as loss normalization

    assert dw_ji.shape == w_ji.shape, "dw_ji shape was: {}. Expected: {}".format(dw_ji.shape, w_ji.shape)

    #Finding gradient between hidden layer H and output layer 
    dw_kj = np.dot(delta_k.T, H)
    dw_kj = dw_kj / normalization_factor # Normalize gradient equally as loss normalization

    assert dw_kj.shape == w_kj.shape, "dw_kj shape was: {}. Expected: {}".format(dw_kj.shape, w_kj.shape)

    if should_check_gradient:
        check_gradient(X, H, w_ji, 1e-2,  dw_ji, 'hidden')
        check_gradient(H, targets, w_kj, 1e-2,  dw_kj, 'out')

    #Updating weights
    w_ji = w_ji - learning_rate*dw_ji
    w_kj = w_kj - learning_rate*dw_kj

    #Momentum
    if should_add_momentum:
      global w_ji_m, w_kj_m
      w_ji = w_ji - momentum*w_ji_m
      w_kj = w_kj - momentum*w_kj_m   
      w_ji_m = dw_ji
      w_kj_m = dw_kj

    return w_ji, w_kj

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
hidden_layer_size = 64

#For implementing momentum
momentum = 0.9
w_ji_m = 0
w_kj_m = 0

#Flags 
should_gradient_check = False
should_shuffle = True
should_improve_sigmoid = True
should_normal_weights = True
should_add_momentum = True

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
      w_kj = np.random.normal(0,1/np.sqrt(hidden_layer_size),(Y_t.shape[1], hidden_layer_size))
    else:
      w_ji = np.random.uniform(-1,1,(hidden_layer_size, X_t.shape[1]))
      w_kj = np.random.uniform(-1,1,(Y_t.shape[1], hidden_layer_size))

    #Training
    for e in range(max_epochs): # Epochs
        for i in tqdm.trange(num_batches):
            X_batch = X_train[i*batch_size:(i+1)*batch_size]
            Y_batch = Y_train[i*batch_size:(i+1)*batch_size]
            H_batch = forward_to_hidden(X_batch,w_ji)
            w_ji, w_kj = gradient_descent(H_batch, X_batch, Y_batch, w_ji, w_kj, learning_rate, should_gradient_check)
            if i % check_step == 0:
                # Loss
                TRAIN_LOSS.append(cross_entropy_loss(X_train, Y_train, w_ji, w_kj))
                TEST_LOSS.append(cross_entropy_loss(X_test, Y_test, w_ji, w_kj))
                VAL_LOSS.append(cross_entropy_loss(X_val, Y_val, w_ji, w_kj))
                
                # Accuracy
                TRAIN_ACC.append(calculate_accuracy(X_train, Y_train, w_ji, w_kj))
                VAL_ACC.append(calculate_accuracy(X_val, Y_val, w_ji, w_kj))
                TEST_ACC.append(calculate_accuracy(X_test, Y_test, w_ji, w_kj))
                if should_early_stop(VAL_LOSS):
                    print(VAL_LOSS[-4:])
                    print("early stopping at epoch {}".format(e))
                    return w_ji, w_kj
        print("Epoch: {} | Accuracy: {} | Loss :{}".format(e+1, round(VAL_ACC[-1],2), round(VAL_LOSS[-1],2)))
        X_t, Y_t = shuffle(X_train, Y_train)
    return w_ji, w_kj


w_ji, w_kj = train_loop()

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








