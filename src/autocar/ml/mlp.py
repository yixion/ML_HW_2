import numpy as np


class MLPRegressor:
    def __init__(
            self,
            input_size,
            hidden_sizes,
            seed=42
            ):
        self.rng = np.random.default_rng(seed)
        self.W = []
        self.layer_outputs = []
        self.hidden_sizes = tuple(hidden_sizes)
        self.layer_sizes = (input_size, *self.hidden_sizes, 1)
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
        layer_input = np.insert(features, 0,-1, axis=1)
        for i in range(len(self.hidden_sizes)):
            layer_output = layer_input @ self.W[i]
            layer_output = self.sigmoid(layer_output)
            self.layer_outputs.append(layer_output)
            layer_input = np.insert(layer_output,0,-1,axis=1)
        layer_output = layer_input @ self.W[len(self.hidden_sizes)]
        self.layer_outputs.append(layer_output)
        return layer_output
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
    print(transform)
    print(2.0*(1/(1+np.exp((-1.0)*features)))+0.5)
