from matplotlib import pyplot as plt
import numpy as np

def get_image(image_path):
    image = plt.imread(image_path)
    return image/255.0


def show_image(image):
    plt.imshow(image)
    plt.show()

def save_image(image, image_path):
    plt.imsave(image_path, image)


def error(original_image: np.ndarray, clustered_image: np.ndarray) -> float:
    # Returns the Mean Squared Error between the original image and the clustered image
    return np.square(original_image-clustered_image).mean()
def plot_mse(X,K):
    val = ["K="+str(i) for i in K]
    plt.bar([i for i in range(1,len(K)+1)], X)
    plt.xlabel("K- Values")
    plt.ylabel("MSE")
    plt.xticks([r + 0.03 for r in range(1,len(K)+1)], val,fontsize ='10')
    plt.title("MSE with respect to K")
    plt.show()
    plt.savefig('MSE.png')