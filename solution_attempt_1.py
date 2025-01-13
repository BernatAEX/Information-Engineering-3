import numpy as np
import os 

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
    




# Example usage:
if __name__ == "__main__":
    c = np.array([0,0,0,1,0,0,0,0,0,0,0,0,1,0,1,1]) # connections vector
    seed = np.array([1,0,1,0,0,1,1,1,0,1,0,1,0,1,1,1]) #initial state   
    image_names = os.listdir("image_folder")
    
    total_code = []
    for a in image_names:
        total_code.append(load_npy_file(a))

    total_code = np.array(total_code)
    print(total_code)
    
