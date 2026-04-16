import numpy as np

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def jiaying_ssim_compare(gt, pred):
    '''
    Compares two tensors of images using SSIM. 
    To keep a consistent range of pixel values, values in the 
    predicted stiffness map were clamped to the max and min of the ground
    truth. Using formula from Jiaying's paper.
    
    Input:
    - gt - ground truth image
    - pred - predicted image
    '''
    
    clamped_pred = torch.clamp(pred,min=torch.min(gt),max = torch.max(gt)).numpy()
    
    # Ensure correct type
    if isinstance(gt, torch.Tensor):
        gt = gt.numpy()
    elif isinstance(gt, np.ndarray):
        gt = gt
    else:
        print('Ground Truth type is not a Tensor nor an Numpy array')
    
    #ssim_score, dif = ssim(gt, clamped_pred, full=True, data_range=np.max(gt)-np.min(gt))
    
    range_values = np.max(gt)-np.min(gt)
    gt_mean = np.mean(gt)
    pred_mean = np.mean(clamped_pred)
    gt_var = np.std(gt)**2
    pred_var = np.std(clamped_pred)**2
    pred_gt_cov = np.cov(gt.flatten(),clamped_pred.flatten())[0,1]
    
    c1 = (0.01**2*range_values)
    c2 =(0.03**2*range_values)
    
    num1 = (2*gt_mean*pred_mean+c1)
    num2 = (2*pred_gt_cov+c2)
    denom1 = (gt_mean**2+pred_mean**2+c1)
    denom2 = (pred_var+gt_var+c2)
    
    SSIM = (num1*num2)/(denom1*denom2)
    return SSIM


def generate_circle_matrix(size, radius=None, center=None):
    """
    Generate a binary matrix with a circle inside.
    
    Parameters:
        size (int): Width and height of the square matrix
        radius (int, optional): Radius of the circle (defaults to size // 2 - 1)
        center (tuple, optional): (x, y) center of the circle (defaults to center of matrix)
    
    Returns:
        np.ndarray: 2D matrix with 1s inside the circle, 0s outside
    """
    if radius is None:
        radius = size // 2 - 1
    if center is None:
        center = (size // 2, size // 2)

    y, x = np.ogrid[:size, :size]
    dist_from_center = (x - center[0])**2 + (y - center[1])**2
    circle_mask = dist_from_center <= radius**2

    return circle_mask.astype(int)

def generate_square_matrix(size, side_length=None):# Parameters
    background_value = 3000
    inclusion_value = 6000
    if side_length is None:
        side_length = size // 2 - 1
    # Create image and mask with background values
    img = np.ones((size, size)) * background_value

    # Define square inclusion
    start = size // 2 - side_length // 2
    end = start + side_length
    img[start:end, start:end] = inclusion_value

    return img.astype(int)


def cnr_compare(img, gt_mask, visualize = False):
    '''
    Calculates Contrast-to-Noise Ratio (CNR) given the test image 
    and the ground truth mask of inclusion and background material.
    Using formula from Jiaying's paper.
    
    Inputs:
    - img - Image to calculate the CNR
    - gt_mask - Ground truth mask which shows inclusion at 6000 Pa and background at 3000 Pa
    - visualize: Boolean flag to display images and masks for visual inspection.
    
    '''
    if isinstance(img, torch.Tensor):
        img_np = img.numpy()
    elif isinstance(img, np.ndarray):
        img_np = img
    else:
        print('Image type is not a Tensor nor an Numpy array')
        
        
    
    # Filter values based on ground truth mask
    inclusion_mus = img_np[gt_mask > 4500]
    background_mus = img_np[gt_mask <= 4500]
    
    inclusion_mean = np.mean(inclusion_mus)
    inclusion_var = np.std(inclusion_mus)**2
    background_mean = np.mean(background_mus)
    background_var = np.std(background_mus)**2
    
    if visualize:
        inclusion_mask = (gt_mask > 4500).astype(int)
        inclusion_mus = img*inclusion_mask

        background_mask = ~inclusion_mask
        background_mus = img*background_mask
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap='gray')
        plt.title("Original Image")
        plt.axis("off")
        
        # Plot the inclusion mask
        plt.subplot(1, 3, 2)
        plt.imshow(inclusion_mask, cmap='jet', alpha=0.5)
        plt.title("Inclusion Mask")
        plt.axis("off")
        
        # Plot the background mask
        plt.subplot(1, 3, 3)
        plt.imshow(background_mask, cmap='jet', alpha=0.5)
        plt.title("Background Mask")
        plt.axis("off")
        
        plt.tight_layout()
        plt.show()
    # Debug prints
#     print("Inclusion mean:", inclusion_mean)
#     print("Background mean:", background_mean)
#     print("Inclusion variance:", inclusion_var)
#     print("Background variance:", background_var)
        
    
    return np.sqrt(2*(inclusion_mean-background_mean)**2/(inclusion_var+background_var))

def mae_compare(gt, pred):
    '''
    Compares two tensors of images using MAE. 
    
    Input:
    - gt - ground truth image
    - pred - predicted image
    '''
    
    # Ensure correct type
    if isinstance(gt, torch.Tensor):
        gt = gt.numpy()
    elif isinstance(gt, np.ndarray):
        gt = gt
    else:
        print('Ground Truth type is not a Tensor nor an Numpy array')
        
    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    elif isinstance(pred, np.ndarray):
        pred = pred
    else:
        print('Prediction type is not a Tensor nor an Numpy array')
        
    mae = np.mean(np.abs((gt - pred)))
    return mae

def mse_compare(gt, pred):
    '''
    Compares two tensors of images using MSE. 
    
    Input:
    - gt - ground truth image
    - pred - predicted image
    '''
    
    # Ensure correct type
    if isinstance(gt, torch.Tensor):
        gt = gt.numpy()
    elif isinstance(gt, np.ndarray):
        gt = gt
    else:
        print('Ground Truth type is not a Tensor nor an Numpy array')
        
    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    elif isinstance(pred, np.ndarray):
        pred = pred
    else:
        print('Prediction type is not a Tensor nor an Numpy array')
    return mse(gt, pred)

def metrics_unit_test():
    """
    A unit test function to evaluate the performance of various error metrics 
    (MSE, MAE, SSIM, and CNR) on two synthetic images.

    This function performs the following steps:
    1. Defines two synthetic 7x7 images (image_1 and image_2) with small differences 
       between them to test the metrics.
    2. Computes the Mean Squared Error (MSE) between the two images both manually 
       and using the `mse_compare` function.
    3. Computes the Mean Absolute Error (MAE) between the two images both manually 
       and using the `mae_compare` function.
    4. Computes the Structural Similarity Index (SSIM) between the two images both manually 
       and using the `jiaying_ssim_compare` function.
    5. Computes the Contrast-to-Noise Ratio (CNR) between the two images using mean and 
       standard deviation values, and compares it using the `cnr_compare` function.
    
    The results of each metric are printed, and comparisons are made between the 
    manually calculated values and those computed by the respective functions.
    
    Note:
    - The `jiaying_ssim_compare` function expects tensor inputs.
    - The `cnr_compare` function is assumed to calculate CNR based on a method from 
      Jiaying's paper.

    Prints:
    - MSE (hardcoded and from `mse_compare()`)
    - MAE (hardcoded and from `mae_compare()`)
    - SSIM (hardcoded and from `jiaying_ssim_compare()`)
    - CNR (from `cnr_compare()`)
    """
        
    # Synthetic images for testing
    # Define image1
    image_1 = np.array([[1, 2, 3, 4, 5, 6, 7],
                       [8, 9, 10, 11, 12, 13, 14],
                       [15, 16, 17, 18, 19, 20, 21],
                       [22, 23, 24, 25, 26, 27, 28],
                       [29, 30, 31, 32, 33, 34, 35],
                       [36, 37, 38, 39, 40, 41, 42],
                       [43, 44, 45, 46, 47, 48, 49]], dtype=np.float32)

    # Define image2 (slightly different value in the last element)
    image_2 = np.array([[1, 2, 3, 4, 5, 6, 7],
                       [8, 9, 10, 11, 12, 13, 14],
                       [15, 16, 17, 18, 19, 20, 21],
                       [22, 23, 24, 25, 26, 27, 28],
                       [29, 30, 31, 32, 33, 34, 35],
                       [36, 37, 38, 39, 40, 41, 42],
                       [43, 44, 45, 46, 47, 48, 50]], dtype=np.float32)

    # Testing MSE
    print("Hard Coded MSE:", np.mean(np.abs(image_1 - image_2)**2))
    print('mse_compare() Results: ', mse_compare(image_1, image_2))

    # Testing MAE
    mae_value = np.mean(np.abs(image_1 - image_2))
    print("Hard Coded MAE:", mae_value)
    print('mae_compare() Results: ', mae_compare(image_1, image_2))

    # Testing SSIM
    ssim_value = ssim(image_1, image_2, data_range=50)
    print("Hard Coded SSIM:", ssim_value)
    print('jiaying_ssim_compare() Results: ', jiaying_ssim_compare(torch.tensor(image_1), torch.tensor(image_2)))

    # Testing CNR (Contrast-to-Noise Ratio)
    mean_image_1 = np.mean(image_1)
    mean_image_2 = np.mean(image_2)
    std_image_1 = np.std(image_1)
    std_image_2 = np.std(image_2)

    # Compute CNR as the ratio of contrast to noise
    print('cnr_compare() Results: ', cnr_compare(image_1, image_2))
    print('CNR follows calculation from Jiaying paper')


def apply_SSIM_error_metrics(model, test_loader, device):
    """
    Evaluates a model's performance on a test dataset using various error metrics: SSIM, MSE, MAE, and Jiaying SSIM.

    This function processes each batch of test data, computes predictions from the model, and calculates the following metrics:
    1. **SSIM (Structural Similarity Index)**: Measures the similarity between the ground truth and predicted images.
    2. **MSE (Mean Squared Error)**: Measures the squared difference between the ground truth and predicted images.
    3. **MAE (Mean Absolute Error)**: Measures the absolute difference between the ground truth and predicted images.
    4. **Jiaying SSIM**: A variant of SSIM used for comparing the ground truth and predicted images.

    The results for each image in the batch are collected and returned as a list of dictionaries containing the metrics.

    Args:
        model (torch.nn.Module): The model to be evaluated. It should accept batches of data and produce predictions.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset, providing batches of data.
        device (torch.device): The device (CPU or GPU) to which the model and data should be moved for evaluation.

    Returns:
        list of dict: A list of dictionaries, where each dictionary contains the evaluation metrics for a single image.
                      Each dictionary includes:
                      - 'index': The unique identifier for the image.
                      - 'SSIM': The SSIM value between the ground truth and predicted image.
                      - 'MSE': The MSE value between the ground truth and predicted image.
                      - 'MAE': The MAE value between the ground truth and predicted image.
                      - 'Jiaying_SSIM': The Jiaying SSIM value between the ground truth and predicted image.

    Example:
        results = apply_SSIM_error_metrics(model, test_loader, device)

    Notes:
        - The function uses `torch.no_grad()` to avoid computing gradients, improving memory usage and speed during inference.
        - The model is assumed to output predictions of the same shape as the ground truth images.
        - `tqdm` is used to provide a progress bar for processing batches.
    """
    results = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Processing batches", unit="batch"):
            wave, gt, mfre, fov, index = batch
            wave = wave.permute(4, 0, 1, 2, 3)

            pred = model(wave.to(device))
            for i in range(wave.shape[0]):  # loop over each image in the batch
                pred_np = pred[i].squeeze().cpu().numpy()
                gt_np = gt[i].squeeze().cpu().numpy()
                idx = index[i].item() if isinstance(index[i], torch.Tensor) else index[i]

                ssim_value = ssim(gt_np, pred_np, data_range=np.max(gt_np)-np.min(gt_np))
                mse = mse_compare(gt_np, pred_np)
                mae = mae_compare(gt_np, pred_np)
                jiaying_ssim = jiaying_ssim_compare(torch.from_numpy(gt_np), torch.from_numpy(pred_np))

                results.append({
                'index': idx,
                'SSIM': ssim_value,
                'MSE': mse,
                'MAE': mae,
                'Jiaying_SSIM': jiaying_ssim,
            })
    
    return results
        
    return SSIM

def plot_error_metrics_histograms(df):
    """
    Plots histograms for SSIM, MSE, MAE, and Jiaying SSIM from the provided dataframe.

    Args:
        df (pandas.DataFrame): A dataframe containing the columns 'SSIM', 'MSE', 'MAE', and 'Jiaying_SSIM'.
                                      The function will plot histograms for each of these columns.

    Returns:
        None: Displays the histograms using Matplotlib.
    """
    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Create a 2x2 grid of subplots
    axes = axes.flatten()  # Flatten the axes array to make it easier to access

    # Plot SSIM
    sns.histplot(df_sorted['SSIM'], bins=30, kde=True, color='skyblue', ax=axes[0])
    axes[0].set_title('SSIM Histogram')
    axes[0].set_xlabel('SSIM')
    axes[0].set_ylabel('Frequency')

    # Plot MSE
    sns.histplot(df_sorted['MSE'], bins=30, kde=True, color='lightgreen', ax=axes[1])
    axes[1].set_title('MSE Histogram')
    axes[1].set_xlabel('MSE')
    axes[1].set_ylabel('Frequency')

    # Plot MAE
    sns.histplot(df_sorted['MAE'], bins=30, kde=True, color='salmon', ax=axes[2])
    axes[2].set_title('MAE Histogram')
    axes[2].set_xlabel('MAE')
    axes[2].set_ylabel('Frequency')

    # Plot Jiaying SSIM
    sns.histplot(df_sorted['Jiaying_SSIM'], bins=30, kde=True, color='lightcoral', ax=axes[3])
    axes[3].set_title('Jiaying SSIM Histogram')
    axes[3].set_xlabel('Jiaying SSIM')
    axes[3].set_ylabel('Frequency')

    # Adjust the layout to make space for titles and labels
    plt.tight_layout()
    plt.show()
