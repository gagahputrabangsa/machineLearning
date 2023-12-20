import numpy as np

class MLP2Layer:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Inisialisasi bobot dan bias secara acak untuk lapisan tersembunyi
        self.weights_hidden = np.random.randn(self.input_dim, self.hidden_dim)
        self.bias_hidden = np.zeros((1, self.hidden_dim))

        # Inisialisasi bobot dan bias secara acak untuk lapisan keluaran
        self.weights_output = np.random.randn(self.hidden_dim, self.output_dim)
        self.bias_output = np.zeros((1, self.output_dim))

    def forward(self, X):
        # Feedforward untuk lapisan tersembunyi
        hidden_layer_input = np.dot(X, self.weights_hidden) + self.bias_hidden
        hidden_layer_output = self.sigmoid(hidden_layer_input)

        # Feedforward untuk lapisan keluaran
        output_layer_input = np.dot(hidden_layer_output, self.weights_output) + self.bias_output
        output_layer_output = self.sigmoid(output_layer_input)

        return output_layer_output

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, X, y, epochs, learning_rate, momentum=0.0):
        prev_weight_hidden_update = np.zeros_like(self.weights_hidden)
        prev_bias_hidden_update = np.zeros_like(self.bias_hidden)
        prev_weight_output_update = np.zeros_like(self.weights_output)
        prev_bias_output_update = np.zeros_like(self.bias_output)

        for epoch in range(epochs):
            # Forward pass
            hidden_layer_input = np.dot(X, self.weights_hidden) + self.bias_hidden
            hidden_layer_output = self.sigmoid(hidden_layer_input)
            output_layer_input = np.dot(hidden_layer_output, self.weights_output) + self.bias_output
            output_layer_output = self.sigmoid(output_layer_input)

            # Menghitung error
            output_error = y - output_layer_output
            output_delta = output_error * self.sigmoid_derivative(output_layer_output)

            # Menghitung gradien untuk bobot dan bias lapisan keluaran
            weights_output_gradient = np.dot(hidden_layer_output.T, output_delta)
            bias_output_gradient = np.sum(output_delta, axis=0, keepdims=True)

            # Menghitung error untuk lapisan tersembunyi
            hidden_error = np.dot(output_delta, self.weights_output.T)
            hidden_delta = hidden_error * self.sigmoid_derivative(hidden_layer_output)

            # Menghitung gradien untuk bobot dan bias lapisan tersembunyi
            weights_hidden_gradient = np.dot(X.T, hidden_delta)
            bias_hidden_gradient = np.sum(hidden_delta, axis=0, keepdims=True)

            # Update bobot dan bias menggunakan gradien, learning rate, dan momentum
            weight_hidden_update = (learning_rate * weights_hidden_gradient) + (momentum * prev_weight_hidden_update)
            bias_hidden_update = (learning_rate * bias_hidden_gradient) + (momentum * prev_bias_hidden_update)
            weight_output_update = (learning_rate * weights_output_gradient) + (momentum * prev_weight_output_update)
            bias_output_update = (learning_rate * bias_output_gradient) + (momentum * prev_bias_output_update)

            self.weights_output += weight_output_update
            self.bias_output += bias_output_update
            self.weights_hidden += weight_hidden_update
            self.bias_hidden += bias_hidden_update

            # Simpan gradien saat ini untuk digunakan dalam iterasi berikutnya
            prev_weight_hidden_update = weight_hidden_update
            prev_bias_hidden_update = bias_hidden_update
            prev_weight_output_update = weight_output_update
            prev_bias_output_update = bias_output_update

    def predict(self, X):
        return self.forward(X)

    def display_weights(self):
        print("Bobot Lapisan Input ke Hidden:")
        print(self.weights_hidden)
        print("\nBobot Lapisan Hidden ke Output:")
        print(self.weights_output)
        
    def calculate_total_error(self, X, y):
        total_error = 0
        for i in range(len(X)):
            prediction = self.forward(X[i:i+1])
            error = np.sum(0.5 * (y[i:i+1] - prediction) ** 2)
            total_error += error
        return total_error / len(X)

if __name__ == "__main__":
    input_dim = 3
    hidden_dim = 6
    output_dim = 2

    mlp = MLP2Layer(input_dim, hidden_dim, output_dim)

    while True:
        print("------[MLP2Layer]----------")
        print("| 1. Buat MLP2Layer      |")
        print("| 2. Training           |")
        print("| 3. Prediksi           |")
        print("| 4. Tampilkan bobot    |")
        print("| 5. Keluar             |")
        print("---------------------------")
        choice = input("Pilih? ")

        if choice == "1":
            input_dim = int(input("Jumlah input: "))
            hidden_dim = int(input("Jumlah neuron hidden layer: "))
            output_dim = int(input("Jumlah neuron output: "))
            mlp = MLP2Layer(input_dim, hidden_dim, output_dim)
            print(f"Buat objek MLP2Layer({input_dim}, {hidden_dim}, {output_dim}) telah dibuat!")

        elif choice == "2":
            problem_type = input("Problem (and, or, xor): ").lower()
            if problem_type not in ['a', 'o', 'x']:
                print("Problem yang Anda masukkan tidak valid.")
            else:
                X_train = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
                if problem_type == 'a':
                    y_train = np.array([[0, 1], [1, 0], [1, 0], [0, 1]])
                elif problem_type == 'o':
                    y_train = np.array([[0, 1], [1, 0], [1, 0], [1, 0]])
                elif problem_type == 'x':
                    y_train = np.array([[1, 0], [1, 0], [1, 0], [0, 1]])

                learning_rate = float(input("Masukkan learning rate: "))
                momentum = float(input("Masukkan momentum: "))
                epochs = int(input("Masukkan jumlah epoch: "))
                
                print("---- INFO TRAINING ----")
                for epoch in range(epochs):
                    error = mlp.calculate_total_error(X_train, y_train)
                    print(f"Epoch={epoch+1}, error={error}")

                    mlp.train(X_train, y_train, 1, learning_rate, momentum)

                print("\nTraining selesai!\n")

        elif choice == "3":
            input_data = []
            for i in range(input_dim):
                input_value = float(input(f"Input {i+1}? "))
                input_data.append(input_value)

            input_data = np.array(input_data).reshape(1, -1)
            prediction = mlp.predict(input_data)
            print("Hasil prediksi:")
            print(prediction)

        elif choice == "4":
            mlp.display_weights()

        elif choice == "5":
            print("Keluar dari program.")
            break

        else:
            print("Pilihan tidak valid. Silakan pilih lagi.")
