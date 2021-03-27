import mnist_loader
import network
import mnist_svm
from time import sleep

def main():

    print("Ejecutando main...")
    print("Entrenando duro...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)
    print("Entrenamiento terminado...")

    print("Validando data...")
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
    mnist_svm.svm_baseline()
    sleep(5) # Time in seconds
    print("Proceso terminado...")
    pass

main()