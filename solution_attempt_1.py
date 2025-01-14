import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image as im
from tqdm import tqdm

from collections import defaultdict

def load_npy_file(file_path):
    
    """Loads and returns the array from a .npy file.

    :param file_path: Path to the .npy file
    :return: The array stored in the .npy file
    """
    try:
        array = np.load(file_path)
        #print("Array loaded successfully!")
        return array
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None
    

def LFSR(N,c,seed): 
    LFSR_output = np.zeros(N)
    m = np.size(c)
    state = seed
    for i in range(0,N): 
        next_bit = np.mod(np.sum(c*state),2)
        LFSR_output[i] = state[m-1]
        state = np.concatenate((np.array([next_bit]) , state[0:m-1]))
    return LFSR_output  

def entropy(frequencies) -> float:
    total_count = sum(frequencies.values())
    entropy_value = 0

    for _, freq in frequencies.items():
        p_i = freq / total_count
        if p_i > 0:
            entropy_value -= p_i * np.log2(p_i)

    return entropy_value

def compute_entropy(code):
    
    r_freq =defaultdict(int)
    g_freq =defaultdict(int)
    b_freq =defaultdict(int)

    for pixel in code[0]:
        r, g, b = pixel

        rint = int(r)
        gint = int(g)
        bint = int(b)

        r_freq[rint] += 1
        g_freq[gint] += 1
        b_freq[bint] += 1

    entropy_r = entropy(r_freq)
    entropy_g = entropy(g_freq)
    entropy_b = entropy(b_freq)

    # Sum up the entropies of each channel
    total_entropy = (entropy_r + entropy_g + entropy_b)/3
    print(total_entropy)
    return total_entropy
    
def genSeed(num):
    if 0 <= num <= 2**16 - 1:
        binary = format(num, '016b')
        arr16b = np.array([int(bit) for bit in binary], dtype=np.int8)
        return arr16b
    else:
        raise ValueError("Out of range int")

def xor_func(im_1,im_2):

    output = np.zeros(len(im_1))
    if len(im_1)!=len(im_2):
        print("not the same length")
        return
    np.logical_xor(im_1,im_2)
    
    for i in range(len(im_1)):
        if im_1[i]==im_2[i]:
            output[i]=0
        else:
            output[i]=1

    return output

def visualize_image(image_bits):
    image_rec_plane = np.zeros((200,200,3),dtype=np.uint8)

    # From binary to image
    image_binary_decrypt_plane = np.reshape(image_bits, (200*200*8,3)) 
    for i_plane in range(0,3):
        ctr = 0
        for i in range(0,200):
            for j in range(0,200):
                image_rec_plane[i,j,i_plane] = np.sum(image_binary_decrypt_plane[ctr:ctr+8,i_plane]*np.array([128,64,32,16,8,4,2,1]))
                ctr += 8

    # Recovering the image from the array of YCbCr
    image_rec = im.fromarray(image_rec_plane) 
    
    return image_rec

def xor_visualization(total_code):
    f, axarr = plt.subplots(1,3)
    axarr[0].imshow(visualize_image(np.logical_xor(total_code[0],total_code[1])))
    axarr[1].imshow(visualize_image(np.logical_xor(total_code[0],total_code[2])))
    axarr[2].imshow(visualize_image(np.logical_xor(total_code[1],total_code[2])))
    plt.show()

    return None

def detect_16(xor_output):
    
    ref= np.zeros(16)
    idx_list=[]
    for i in range(len(xor_output)-16):
        check = xor_output[i:i+1]
        if (check == ref).all:
            print("similarity found")
            print(i)
            idx_list.append(idx_list)
    return idx_list       
def detect_segment16(lis):
    for i in range(len(lis) - 15):
        if (lis[i:i + 16] == np.zeros(16)).all:
            return i
    return -1


def decode(coded_image_o, state_equation, seed, reduction):
    coded_image = coded_image_o[:int(len(coded_image_o)/(reduction**2))]
    cypher_rec = LFSR(np.size(coded_image), state_equation,seed)

    image_binary_decrypt = np.mod(coded_image+cypher_rec,2) 
    image_rec_plane = np.zeros((int(200/reduction),int(200/reduction),3),dtype=np.uint8)
    image_binary_decrypt_plane = np.reshape(image_binary_decrypt, (int(200/reduction)*int(200/reduction)*8,3)) 
    for i_plane in range(0,3):     
        ctr = 0
        for i in range(0, int(200/reduction)):
            for j in range(0, int(200/reduction)):
                image_rec_plane[i,j,i_plane] = (np.sum(image_binary_decrypt_plane[ctr:ctr+8,i_plane]*np.array([128,64,32,16,8,4,2,1]))) 
                ctr += 8

    entropy_im = compute_entropy(image_rec_plane)
    return image_rec_plane, entropy_im

# Example usage:
if __name__ == "__main__":
    c = np.array([0,0,0,1,0,0,0,0,0,0,0,0,1,0,1,1]) # connections vector
    seed = np.array([1,0,1,0,0,1,1,1,0,1,0,1,0,1,1,1]) #initial state   
    image_names = os.listdir("image_folder")


    
    total_code = []
    for a in image_names:
        total_code.append(load_npy_file(a))

    total_code = np.array(total_code)
    
    output = xor_func(total_code[0], total_code[2])
    print(output)
    position = detect_segment16(output)
    print(position)

    potential_seed = total_code[0][:16]
    potential_seed = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1])
    #potential_seed = np.array([0, 0, 0, 0, 0, 0, 1, 1,0, 1, 1, 1, 0, 1, 0, 1])
    print(potential_seed)

    output_image, _ = decode(total_code[2], c, potential_seed, 1)

    plt.imshow(output_image)
    plt.show()


    min_entropy = np.inf
    opt_seed = None
    for i in tqdm(range(1,int(2**16)), position=0, leave=True):
        try_seed = genSeed(i)

        _, entropy_try = decode(total_code[0], c, try_seed, reduction=40)
        if entropy_try < min_entropy:
            min_entropy = entropy_try
            opt_seed = try_seed
       
        #os.system('cls')
    print(opt_seed)
    print(min_entropy)