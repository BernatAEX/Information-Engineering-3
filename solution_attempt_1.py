import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image as im

def load_npy_file(file_path):
    """
    Loads and returns the array from a .npy file.

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

def entropy(frequencies: Dict[int, float]) -> float:
    total_count = sum(frequencies.values())
    entropy_value = 0

    for _, freq in frequencies.items():
        p_i = freq / total_count
        if p_i > 0:
            entropy_value -= p_i * log2(p_i)

    return entropy_value

    
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

def visualization(image):
    image_rec_plane = np.zeros((200,200,3),dtype=np.uint8)

    # From binary to image
    image_binary_decrypt_plane = np.reshape(image, (200*200*8,3)) 
    for i_plane in range(0,3):
        ctr = 0
        for i in range(0,np.size(np.size(image),0)):
            for j in range(0, np.size(np.size(image),1)):
                image_rec_plane[i,j,i_plane] = (np.sum(image_binary_decrypt_plane[ctr:ctr+8,i_plane]*np.array([128,64,32,16,8,4,2,1]))) 
                ctr += 8

    # Recovering the image from the array of YCbCr
    image_rec = im.fromarray(image_rec_plane) 
    # Plot the image 
    plt.imshow(image_rec)
    #plt.show()
    
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
    print(potential_seed)
    
    
    
