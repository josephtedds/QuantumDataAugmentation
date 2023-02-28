# This code is part of Qiskit.
#
# (C) Copyright IBM 2020s.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# This code has been altered from the original, which can be found at:
# https://github.com/qiskit-community/QuantumBlur

import math
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import qiskit

from myrtle_core import (
    Crop,
    Cutout,
    FlipLR,
    Transform,
    pad,
    preprocess,
    transpose,
)
from myrtle_torch_backend import cifar10


def _image2heights(image: np.ndarray):
    """
    Converts an rgb image into a list of three height dictionaries, one for
    each colour channgel.
    """
    _, Lx, Ly = image.shape
    heights = [
        {(x, y): image[j, x, y] for x in range(Lx) for y in range(Ly)}
        for j in range(3)
    ]
    return heights


def _get_size(height):
    """
    Determines the size of the grid for the given height map.
    """
    Lx = 0
    Ly = 0
    for (x, y) in height:
        Lx = max(x + 1, Lx)
        Ly = max(y + 1, Ly)
    return Lx, Ly


def _heights2image(heights):
    """
    Constructs an image from a set of three height dictionaries, one for each
    colour channel.
    """
    Lx, Ly = _get_size(heights[0])
    image = np.zeros((3, Lx, Ly), dtype=int)

    for i, i_heights in enumerate(heights):
        i_h_max = max(i_heights.values())
        for x in range(Lx):
            for y in range(Ly):
                if (x, y) in i_heights:
                    image[i, x, y] = int(
                        255 * float(i_heights[x, y]) / i_h_max
                    )
                else:
                    image[i, x, y] = 0
    return image


def normalize(ket):
    """
    Normalizes the given statevector.

    Args:
        ket (list or array_like)

    Returns:
        ket (list or array_like)
    """
    N = 0
    for amp in ket:
        N += amp * amp.conjugate()
    if N == 0:
        ket = [1 / math.sqrt(len(ket)) for i in range(len(ket))]
    else:
        for j, amp in enumerate(ket):
            ket[j] = float(amp) / math.sqrt(N)
    return ket


def height2circuit(height, log=False, eps=1e-2):
    """
    Converts a dictionary of heights (or brightnesses) on a grid into
    a quantum circuit.

    Args:
        height (dict): A dictionary in which keys are coordinates
            for points on a grid, and the values are positive numbers of
            any type.
        log (bool): If given, a logarithmic encoding is used.

    Returns:
        qc (QuantumCircuit): A quantum circuit which encodes the
            given height dictionary.
    """
    # get bit strings for the grid
    Lx, Ly = _get_size(height)
    grid, n = make_grid(Lx, Ly)

    # create required state vector
    state = [0] * (2**n)
    if log:
        # normalize heights
        max_h = max(height.values())
        height = {pos: float(height[pos]) / max_h for pos in height}
        # find minimum (not too small) normalized height
        min_h = min([height[pos] for pos in height if height[pos] > eps])
        # this minimum value defines the base
        base = 1.0 / min_h
    for bitstring in grid:
        (x, y) = grid[bitstring]
        if (x, y) in height:
            h = height[x, y]
            if log:
                state[int(bitstring, 2)] = math.sqrt(
                    base ** (float(h) / min_h)
                )
            else:
                state[int(bitstring, 2)] = math.sqrt(h)
    state = normalize(state)

    # define and initialize quantum circuit
    qc = qiskit.QuantumCircuit(n)
    qc.initialize(state, range(n))
    qc.name = "(" + str(Lx) + "," + str(Ly) + ")"

    return qc


def image2circuits(image, log=False):
    """
    Converts an image to a set of three circuits, with one corresponding to
    each RGB colour channel.
    Args:
        image (Image): An RGB encoded image.
        log (bool): If given, a logarithmic encoding is used.
    Returns:
        circuits (list): A list of quantum circuits encoding the image.
    """

    heights = _image2heights(image)

    circuits = []
    for height in heights:
        circuits.append(height2circuit(height, log=log))

    return circuits


def make_line(length):
    """
    Creates a list of bit strings of at least the given length, such
    that the bit strings are all unique and consecutive strings
    differ on only one bit.

    Args:
        length (int): Required length of output list.

    Returns:
        line (list): List of 2^n n-bit strings for n=⌊log_2(length)⌋
    """

    # number of bits required
    n = int(math.ceil(math.log(length) / math.log(2)))

    # iteratively build list
    line = ["0", "1"]
    for j in range(n - 1):
        # first append a reverse-ordered version of the current list
        line = line + line[::-1]
        # then add a '0' onto the end of all bit strings in the first half
        for j in range(int(float(len(line)) / 2)):
            line[j] += "0"
        # and a '1' for the second half
        for j in range(int(float(len(line)) / 2), int(len(line))):
            line[j] += "1"

    return line


def make_grid(Lx, Ly=None):
    """
    Creates a dictionary that provides bit strings corresponding to
    points within an Lx by Ly grid.

    Args:
        Lx (int): Width of the lattice (also the height if no Ly is
            supplied).
        Ly (int): Height of the lattice if not Lx.

    Returns:
        grid (dict): Dictionary whose values are points on an
            Lx by Ly grid. The corresponding keys are unique bit
            strings such that neighbouring strings differ on only
            one bit.
        n (int): Length of the bit strings

    """
    # set Ly if not supplied
    if not Ly:
        Ly = Lx

    # make the lines
    line_x = make_line(Lx)
    line_y = make_line(Ly)

    # make the grid
    grid = {}
    for x in range(Lx):
        for y in range(Ly):
            grid[line_x[x] + line_y[y]] = (x, y)

    # determine length of the bit strings
    n = len(line_x[0] + line_y[0])

    return grid, n


def circuits2image(circuits, log=False):
    """
    Extracts an image from list of circuits encoding the RGB channels.
    Args:
        circuits (list): A list of quantum circuits encoding the image.
        log (bool): If given, a logarithmic decoding is used.
    Returns:
        image (Image): An RGB encoded image.
    """

    heights = []
    for qc in circuits:
        heights.append(circuit2height(qc, log=log))

    return _heights2image(heights)


def circuit2height(qc, log=False):
    """
    Extracts a dictionary of heights (or brightnesses) on a grid from
    the quantum circuit into which it has been encoded.

    Args:
        qc (QuantumCircuit): A quantum circuit which encodes a height
            dictionary. The name attribute should hold the size of
            the image to be created (as a tuple cast to a string).
        log (bool): If given, a logarithmic decoding is used.

    Returns:
        height (dict): A dictionary in which keys are coordinates
            for points on a grid, and the values are floats in the
            range 0 to 1.
    """

    probs = circuit2probs(qc)
    try:
        # get size from circuit
        size = eval(qc.name)
    except:
        # if not in circuit name, infer it from qubit number
        L = int(2 ** (qc.num_qubits / 2))
        size = (L, L)
    return probs2height(probs, size=size, log=log)


def _kron(vec0, vec1):
    """
    Calculates the tensor product of two vectors.
    """
    new_vec = []
    for amp0 in vec0:
        for amp1 in vec1:
            new_vec.append(amp0 * amp1)
    return new_vec


def circuit2probs(qc):
    """
    Runs the given circuit, and returns the resulting probabilities.
    """

    # separate circuit and initialization
    new_qc = qc.copy()
    new_qc.data = []
    initial_ket = [1]
    for gate in qc.data:
        if gate[0].name == "initialize":
            initial_ket = _kron(initial_ket, gate[0].params)
        else:
            new_qc.data.append(gate)
    # if there was no initialization, use the standard state
    if len(initial_ket) == 1:
        initial_ket = [0] * 2**qc.num_qubits
        initial_ket[0] = 1
    # then run it
    ket = qiskit.quantum_info.Statevector(initial_ket)
    ket = ket.evolve(new_qc)
    probs = ket.probabilities_dict()

    return probs


def probs2height(probs, size=None, log=False):
    """
    Extracts a dictionary of heights (or brightnesses) on a grid from
    a set of probabilities for the output of a quantum circuit into
    which the height map has been encoded.

    Args:
        probs (dict): A dictionary with results from running the circuit.
            With bit strings as keys and either probabilities or counts as
            values.
        size (tuple): Size of the height map to be created. If not given,
            the size is deduced from the number of qubits (assuming a
            square image).
        log (bool): If given, a logarithmic decoding is used.

    Returns:
        height (dict): A dictionary in which keys are coordinates
            for points on a grid, and the values are floats in the
            range 0 to 1.
    """

    # get grid info
    if size:
        (Lx, Ly) = size
    else:
        Lx = int(2 ** (len(list(probs.keys())[0]) / 2))
        Ly = Lx
    grid, _ = make_grid(Lx, Ly)

    # set height to probs value, rescaled such that the maximum is 1
    max_h = max(probs.values())
    height = {(x, y): 0.0 for x in range(Lx) for y in range(Ly)}
    for bitstring in probs:
        if bitstring in grid:
            height[grid[bitstring]] = float(probs[bitstring]) / max_h

    # take logs if required
    if log:
        min_h = min([height[pos] for pos in height if height[pos] != 0])
        alt_min_h = min([height[pos] for pos in height])
        base = 1 / min_h
        for pos in height:
            if height[pos] > 0:
                height[pos] = max(
                    math.log(height[pos] / min_h) / math.log(base), 0
                )
            else:
                height[pos] = 0.0

    return height


def partial_x(qc, fraction):
    for j in range(qc.num_qubits):
        qc.rx(np.pi * fraction, j)


def colours_channel_first(image):
    if image.shape[0] == 3:
        return image
    else:
        return image.transpose((2, 0, 1))


def colours_channel_last(image):
    if image.shape[-1] == 3:
        return image
    else:
        return image.transpose((1, 2, 0))


def blur_image(image, alpha=0.1):
    image_circuits = image2circuits(image)
    for i_circ in image_circuits:
        partial_x(i_circ, alpha)
    return circuits2image(image_circuits)


if __name__ == "__main__":
    DATA_DIR = "./data"
    dataset = cifar10(root=DATA_DIR)

    transforms = [
        partial(transpose, source="NHWC", target="NCHW"),
    ]

    train_set = list(
        zip(
            *preprocess(
                dataset["train"], [partial(pad, border=4)] + transforms
            ).values()
        )
    )

    train_transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]
    transformed_train_set = Transform(train_set, train_transforms)
    transformed_train_set.set_random_choices()

    img1 = dataset["train"]["data"][0]

    img1_transformed = colours_channel_last(
        transformed_train_set.__getitem__(0)[0]
    )
    img1_blurred = colours_channel_last(
        blur_image(colours_channel_first(img1))
    )

    plt.imshow(img1)
    plt.show()
    plt.imshow(img1_blurred)
    plt.show()
    plt.imshow(img1_transformed)
    plt.show()
