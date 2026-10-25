import numpy as np


class MLPRegressor:
    def __init__(
            self,
            input_size,
            hidden_sizes,
            learning_rate=0.01,
            seed=42
            ):
        self.rng = np.random.default_rng(seed)
        self.W = []
        self.layer_outputs = []
        self.hidden_sizes = tuple(hidden_sizes)
        self.layer_sizes = (input_size, *self.hidden_sizes, 1)
        self.l_r = learning_rate
        for i in range(1,len(self.layer_sizes)):
            w = self.rng.uniform(low=-0.5,high=0.5,size=(self.layer_sizes[i-1]+1, self.layer_sizes[i]),)
            for j in range(self.layer_sizes[i]):
                w[0][j] = 0.0
            self.W.append(w)
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    def forward(self,features):
        if (~np.isfinite(features)).any():
            raise ValueError("ple preprocessing your data")
        self.layer_outputs = []
        self.layer_inputs =[]
        layer_input = np.insert(features, 0,-1, axis=1)
        self.layer_inputs.append(layer_input)
        for i in range(len(self.hidden_sizes)):
            layer_output = layer_input @ self.W[i]
            layer_output = self.sigmoid(layer_output)
            self.layer_outputs.append(layer_output)
            layer_input = np.insert(layer_output,0,-1,axis=1)
            self.layer_inputs.append(layer_input)
        layer_output = layer_input @ self.W[len(self.hidden_sizes)]
        self.layer_outputs.append(layer_output)
        return layer_output
    def backward(self,t):
        if (not self.layer_outputs or not self.layer_inputs):
            raise RuntimeError("no use forward function")
        p = self.layer_outputs[-1]
        if (p.shape != t.shape or t.ndim != 2 or not np.isfinite(t).all()):
            raise ValueError("target include invalid value")
        # calculate output delta 
        delta = t-p # (batch_size, 神經元個數)
        self.delta_list = [None]*len(self.W)
        for i in range(len(self.W)-1,-1,-1):
            if i == len(self.W)-1:
                self.delta_list[i] = delta
            else:
                # (batch_size, 神經元個數)@(神經元個數, 下一層的 input dimension) 
                self.delta_list[i] = (
                        self.layer_outputs[i]
                        * (1-self.layer_outputs[i])
                        * (self.delta_list[i+1] @ self.W[i+1][1:,].T)
                        )
                # (batch_size, 下一層的input dimension == 這一層 hidden layer 的神經元個數)
        # weight update
        batch_size = t.shape[0]
        for i in range(len(self.W)):
            # input 的 shape = (batch size, input feature 個數)
            self.W[i] = self.W[i] + self.l_r * self.layer_inputs[i].T@self.delta_list[i]/batch_size
        print((self.layer_inputs[0].T@self.delta_list[0]).shape)
        print(self.W[0].shape)
        
def mean_squared_error(target,predictions):
    if (target.shape != predictions.shape
        or not np.isfinite(target).all()
        or not np.isfinite(predictions).all()
        or target.ndim != 2
        or predictions.ndim != 2):
            raise ValueError("wrong value of mean squared error")
    
    loss = np.mean((target-predictions)**2)    
    return loss


if __name__ == "__main__":
    model = MLPRegressor(
        input_size=1,
        hidden_sizes=(1,),
        seed=42,
    )

    # 第一列是一般 weight，最後一列是 bias。
    model.W[0] = np.array([
        [0.0],
        [1.0],
    ])
    model.W[1] = np.array([
        [-0.5],
        [2.0],
    ])

    features = np.array([
        [0.0],
        [1.0],
    ])
    transform = model.forward(features)
    model.backward(features)
    print(model.delta_list)
