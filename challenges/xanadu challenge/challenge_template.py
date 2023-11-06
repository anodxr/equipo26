### DO NOT CHANGE ANYTHING BELOW THIS LINE

import pennylane as qml
from pennylane import numpy as np

WIRES = 2
LAYERS = 5
NUM_PARAMETERS = LAYERS * WIRES * 3

def variational_circuit(params,hamiltonian):
    """
    This is a template variational quantum circuit containing a fixed layout of gates with variable
    parameters. To be used as a QNode, it must either be wrapped with the @qml.qnode decorator or
    converted using the qml.QNode function.

    The output of this circuit is the expectation value of a Hamiltonian, somehow encoded in
    the hamiltonian argument

    Args:
        - params (np.ndarray): An array of optimizable parameters of shape (30,)
        - hamiltonian (np.ndarray): An array of real parameters encoding the Hamiltonian
        whose expectation value is returned.
    
    Returns:
        (float): The expectation value of the Hamiltonian
    """
    parameters = params.reshape((LAYERS, WIRES, 3))
    qml.templates.StronglyEntanglingLayers(parameters, wires=range(WIRES))
    return qml.expval(qml.Hermitian(hamiltonian, wires = [0,1]))

def optimize_circuit(hamiltonian):
    """Minimize the variational circuit and return its minimum value.
    You should create a device and convert the variational_circuit function 
    into an executable QNode. 
    Next, you should minimize the variational circuit using gradient-based 
    optimization to update the input params. 
    Return the optimized value of the QNode as a single floating-point number.

    Args:
        - params (np.ndarray): Input parameters to be optimized, of dimension 30
        - hamiltonian (np.ndarray): An array of real parameters encoding the Hamiltonian
        whose expectation value you should minimize.
    Returns:
        float: the value of the optimized QNode
    """
        
    hamiltonian = np.array(hamiltonian, requires_grad = False)

    hamiltonian = np.array(hamiltonian,float).reshape((2 ** WIRES), (2 ** WIRES))

    ### WRITE YOUR CODE BELOW THIS LINE
    
    ### Solution Template

    dev = qml.device("default.qubit", wires=WIRES) 

    circuit = qml.QNode(variational_circuit, dev)

    # Write your code to minimize the circuit

    # Initialize the parameters randomly
    params = np.random.uniform(0, 2*np.pi, size=NUM_PARAMETERS)
    print("Params:", params)
    
    # Define the cost function 
    def cost(params):
        return circuit(params, hamiltonian)

    # Use a gradient-based optimizer to minimize the cost function
    opt = qml.GradientDescentOptimizer(stepsize=0.1)
    steps = 400 # Number of optimization steps
    for i in range(steps):
        params = opt.step(cost, params)
        print(f"Step {i+1}: Cost = {cost(params)}")

    # Return the optimized value of the QNode
    return circuit(params, hamiltonian)

# Test case 1
test_input = [
    0.863327072347624, 0.0167108057202516, 0.07991447085492759, 0.0854049026262154,
    0.0167108057202516, 0.8237963773906136, -0.07695947154193797, 0.03131548733285282,
    0.07991447085492759, -0.07695947154193795, 0.8355417021014687, -0.11345916130631205,
    0.08540490262621539, 0.03131548733285283, -0.11345916130631205, 0.758156886827099
]
print("Input 1:", test_input)
expected_output = optimize_circuit(test_input)
print("Output 1:", expected_output)


# Test case 2
test_input = [
    0.32158897156285354,-0.20689268438270836,0.12366748295758379,-0.11737425017261123,
    -0.20689268438270836,0.7747346055276305,-0.05159966365446514,0.08215539696259792,
    0.12366748295758379,-0.05159966365446514,0.5769050487087416,0.3853362904758938,
    -0.11737425017261123,0.08215539696259792,0.3853362904758938,0.3986256655167206
]
print("Input 2:", test_input)
expected_output = optimize_circuit(test_input)
print("Output 2:", expected_output)
