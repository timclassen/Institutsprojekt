# This script provides function for calculating the psnr between two different YUV-Videos or images
import numpy as np
import warnings


def psnr_yuv(vid1, vid2):
    '''
        Calculates the PSNR between two images or videos. This function supports the calculation of errors between yuv-videos and images

        Parameters:
            vid1 (List containing three numpy arrays or numpy array): YUV-video or numpy array. The psnr is calculated between this and the other parameter
            vid2 (List containing three numpy arrays or numpy array): YUV-video or numpy array. The psnr is calculated between this and the other parameter

        Returns:
            psnr (dict): PSNR between the two videos. The dictionary contains the keys "Y", "U", "V" and "YUV" for the PSNR of the luminance, chrominance and the overall psnr
    '''
    if isinstance(vid1, dict) and isinstance(vid2, dict):
        return psnr_yuv_video(vid1, vid2)
    elif isinstance(vid1, np.ndarray) and isinstance(vid2, np.ndarray):
        return psnr_image(vid1, vid2)
    else:
        raise NotImplementedError("Error: The psnr can only be calculated for images or YUV-Videos. Other data formats are not supported yet.")


def psnr_yuv_video(vid1, vid2):
    '''
        Calculates the PSNR between two yuv-videos

        Parameters:
            vid1 (List containing three numpy arrays): YUV-video. The psnr is calculated between this and the other parameter
            vid2 (List containing three numpy arrays): YUV-video. The psnr is calculated between this and the other parameter

        Returns:
            psnr (dict): PSNR between the two videos. The dictionary contains the keys "Y", "U", "V" and "YUV" for the PSNR of the luminance, chrominance and the overall psnr
    '''
    psnr_dict = {"YUV": 0, "Y": 0, "U": 0, "V": 0}

    if not vid1["Y"].dtype == vid2["Y"].dtype:
        warnings.warn("input videos do not have the same bit depth", RuntimeWarning)

    max_pel_value = 255 if vid1["Y"].dtype == np.uint8 and vid2["Y"].dtype == np.uint8 else 1020

    for frame_count in range(vid1["Y"].shape[0]):
        with warnings.catch_warnings():  # suppress divide by zero warnings as the default behaviore of setting the PSNR to inf in case of zero MSE is expected
            warnings.filterwarnings("ignore", message="divide by zero encountered")
            psnr_dict["YUV"] += 10 * np.log10(max_pel_value**2 / np.square(np.subtract(np.concatenate([np.ndarray.flatten(vid1["Y"][frame_count]), np.ndarray.flatten(vid1["U"][frame_count]), np.ndarray.flatten(vid1["V"][frame_count])]), np.concatenate([np.ndarray.flatten(vid2["Y"][frame_count]), np.ndarray.flatten(vid2["U"][frame_count]), np.ndarray.flatten(vid2["V"][frame_count])]))).mean())
            psnr_dict["Y"] += 10 * np.log10(max_pel_value**2 / np.square(np.subtract(vid1["Y"][frame_count], vid2["Y"][frame_count])).mean())
            psnr_dict["U"] += 10 * np.log10(max_pel_value**2 / np.square(np.subtract(vid1["U"][frame_count], vid2["U"][frame_count])).mean())
            psnr_dict["V"] += 10 * np.log10(max_pel_value**2 / np.square(np.subtract(vid1["V"][frame_count], vid2["V"][frame_count])).mean())

    for key in psnr_dict:
        psnr_dict[key] /= vid1["Y"].shape[0]  # Divide by the number of frames to get the average psnr
    
    return psnr_dict


def psnr_image(img1, img2):
    '''
        Calculates the PSNR between two scalar images

        Parameters:
            img1 (numpy array): Scalar image. The psnr is calculated between this and the other parameter
            img2 (numpy array): Scalar image. The psnr is calculated between this and the other parameter

        Returns:
            psnr (dict): PSNR between the two images. The dictionary contains the keys "Y", "U", "V" and "YUV" for the PSNR of the luminance, chrominance and the overall psnr
    '''
    psnr_dict = {}

    if not img1.dtype == img2.dtype:
        warnings.warn("input images do not have the same bit depth", RuntimeWarning)

    max_pel_value = 255 if img1.dtype == np.uint8 and img2.dtype == np.uint8 else 1020

    with warnings.catch_warnings():  # suppress divide by zero warnings as the default behaviore of setting the PSNR to inf in case of zero MSE is expected
        warnings.filterwarnings("ignore", message="divide by zero encountered")
        warnings.filterwarnings("error", message="mean of empty slice")
        psnr_dict["YUV"] = 10 * np.log10(max_pel_value**2 / np.square(np.subtract(img1, img2)).mean())
        psnr_dict["Y"] = psnr_dict["YUV"]
        psnr_dict["U"] = psnr_dict["YUV"]
        psnr_dict["V"] = psnr_dict["YUV"]

    return psnr_dict