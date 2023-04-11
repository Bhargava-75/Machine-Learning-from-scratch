from model import KMeans
from utils import get_image, show_image, save_image, error , plot_mse


def main():
    # get image
    image = get_image('image.jpg')
    img_shape = image.shape

    # reshape image
    image = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    # print(image.shape)
    mse_error = []
    num_clusters = [2,5,10,20,50] 
    
    for k in num_clusters:

        kmeans = KMeans(k)

        # fit model
        kmeans.fit(image)

        # replace each pixel with its closest cluster center
        image_clustered = kmeans.replace_with_cluster_centers(image)

        # Print the error
        mse_err = error(image, image_clustered)
        mse_error.append(mse_err)
        print(f'\nFor k = {k} MSE : {mse_err}\n')

        # reshape image
        image_clustered = image_clustered.reshape(img_shape)
        save_image(image_clustered, f'image_clustered_{k}.jpg')
    
    #Plotting Values
    print(mse_error)
    # plot_mse(mse_error,num_clusters)


if __name__ == '__main__':
    main()
