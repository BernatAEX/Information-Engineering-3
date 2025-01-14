import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image as im
from tqdm import tqdm
from collections import defaultdict

def load_npy_file(file_path):
    try:
        return np.load(file_path)
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None

def LFSR(N, c, seed): 
    LFSR_output = np.zeros(N)
    m = np.size(c)
    state = seed
    for i in range(N): 
        next_bit = np.mod(np.sum(c * state), 2)
        LFSR_output[i] = state[m-1]
        state = np.concatenate(([next_bit], state[:m-1]))
    return LFSR_output  

def genSeed(num):
    if 0 <= num <= 2**16 - 1:
        binary = format(num, '016b')
        return np.array([int(bit) for bit in binary], dtype=np.int8)
    else:
        raise ValueError("Out of range int")

def xor_func(im_1, im_2):
    if len(im_1) != len(im_2):
        print("Not the same length")
        return None
    return np.logical_xor(im_1, im_2).astype(np.int8)

def bit_transition_rate(image, block_size=50):
    h, w, _ = image.shape
    start_x = np.random.randint(0, h - block_size)
    start_y = np.random.randint(0, w - block_size)
    sampled_block = image[start_x:start_x + block_size, start_y:start_y + block_size]

    horizontal_diff = np.abs(sampled_block[:, :-1] - sampled_block[:, 1:]).sum()
    vertical_diff = np.abs(sampled_block[:-1, :] - sampled_block[1:, :]).sum()
    total_transitions = horizontal_diff + vertical_diff

    total_possible_transitions = 2 * block_size * (block_size - 1)
    return total_transitions / total_possible_transitions

def decode_fast(coded_image, state_equation, seed, block_size=50):
    cypher_rec = LFSR(len(coded_image), state_equation, seed)
    image_binary_decrypt = np.mod(coded_image + cypher_rec, 2)

    # Validar el tamaño del arreglo antes de hacer el reshape
    expected_size = 200 * 200 * 3
    if len(image_binary_decrypt) != expected_size:
        raise ValueError(f"Expected array size {expected_size}, but got {len(image_binary_decrypt)}")

    image_decoded = image_binary_decrypt.reshape(200, 200, 3)

    transition_rate = bit_transition_rate(image_decoded, block_size=block_size)
    return image_decoded, transition_rate

if __name__ == "__main__":
    c = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1])
    seed = np.array([1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1])
    image_folder = "image_folder"
    image_names = os.listdir(image_folder)

    total_code = [load_npy_file(os.path.join(image_folder, name)) for name in image_names]
    total_code = np.array(total_code)

    # Verificar el tamaño de la primera imagen cargada
    print(f"Coded image size: {len(total_code[0])}")

    min_transition_rate = float("inf")
    opt_seed = None

    for i in tqdm(range(2**16), desc="Processing"):
        try_seed = genSeed(i)
        try:
            _, transition_rate = decode_fast(total_code[0], c, try_seed, block_size=50)

            if transition_rate < min_transition_rate:
                min_transition_rate = transition_rate
                opt_seed = try_seed
        except ValueError as e:
            print(e)

    print(f"Optimal seed: {opt_seed}")
    print(f"Minimum transition rate: {min_transition_rate}")
