# This file contains helper functions to read yuv from *.yuv files into arrays and writing arrays as valid yuv files to hard drive.
import numpy as np


def read_yuv_video(video_path, width=0, height=0, bit_depth=0, subsampling_scheme="", n_frames=-1):
    '''
        Loads a YUV-video and converts it to an array of numpy arrays, where each of the numpy arrays corresponds to one channel of the YUV-video.

        Parameters:
            video_path (String): Path to the YUV-video file
            width (int): Width of the video
            height (int): Height of the video
            bitdepth (int): Bit depth of the YUV-video
            subsampling_scheme (String): Subsampling used for the chroma channels of the video. Common options are: 444, 422, 420
            n_frames (int): Number of frames to load. The default (-1) loads the whole video, i.e. all frames

        Returns:
            vid (list): List containig a numpy array of shape (frame_number, width, height) for every channel. 
    '''
    assert bit_depth in [0, 8, 10]  # other bit depths are not supported yet

    video_info = video_path.replace("dec_", "").replace("enc_", "").split("/")[-1]

    if width == 0:  # automatically infer width if not given
        width = int(video_info.split("_")[1].split("x")[0])
    if height == 0:  # automatically infer height if not given
        height = int(video_info.split("_")[1].split("x")[1])
    if bit_depth == 0:  # automatically infer bit depth if not given
        try:    
            bit_depth = int(video_info.split("_")[2].replace("bit", ""))
        except Exception:
            bit_depth = 8
    if subsampling_scheme == "":  # automatically infer bit depth if not given
        subsampling_scheme = video_info.split("_")[-1].split(".")[0]
        if not subsampling_scheme[0] == "4":
            subsampling_scheme = "420"

    pix_luma = width * height
    dtype = np.uint16 if bit_depth == 10 else np.uint8
    raw_pix = np.fromfile(video_path, dtype=dtype)  # load the video

    if n_frames == -1:  # Set the number of frames to the length of the video stream
        bandwidth_factor = __get_bandwidth_factor__(subsampling_scheme)
        pix_frame = int(pix_luma * bandwidth_factor * 3)
        n_frames = len(raw_pix) // pix_frame
        assert len(raw_pix) == n_frames * pix_frame

    # convert the video into a usable format
    vid = {"Y": [], "U": [], "V": []}
    subsampling_factors = __get_subsampling_factors__(subsampling_scheme)
    n = 0
    n_prev = 0
    for f in range(n_frames):
        for c in ["Y", "U", "V"]:
            if c != "Y":
                n += pix_luma // subsampling_factors[0] // subsampling_factors[1]
                vid[c].append(raw_pix[n_prev:n].reshape([height // subsampling_factors[0], width // subsampling_factors[1]]))
            else:
                n += pix_luma
                vid[c].append(raw_pix[n_prev:n].reshape([height, width]))
            n_prev = n

    if len(vid["Y"]) == 0:  # Return empty video if the input file is empty
        return {"Y": np.array([], dtype=dtype), "U": np.array([], dtype=dtype), "V": np.array([], dtype=dtype)}

    vid = [np.stack(vid[v]) for v in vid if not v == "metadata"]
    return {"Y": vid[0], "U": vid[1], "V": vid[2]}


def write_yuv_video(video, path, automatic_file_name_extension=True):    
    '''
        Writes a YUV-Video to a file given by path. The video is expected to be an array containing three numpy arrays. Those numpy array should correspond to the Y, U and V channels. Each of those numpy arrays shoul have a shape with (frame, width, height). Note that the width and height of the U and V (chroma) channels do not necessarily need to be equivalent to the width and height of the Y (luma) channel. 

        Parameters:
            video_pat (list): YUV-Video given as list of numpy arrays, with [Y_numpy_array, U_numpy_array, V_numpy_array]
            path (String): Path to where the video is supposed to be saved
            automatic_file_name_extension (bool): Whether the file name should be automatically extended such that width, height, bit depth and subsampling scheme are encoded in the file name
        
        Returns:
            None
    '''
    assert "Y" in video, "Video does not contain luminance channel"

    if "U" not in video:
        video["U"] = np.ones(video["Y"].shape, dtype=video["Y"].dtype) * 128
        video["V"] = np.ones(video["Y"].shape, dtype=video["Y"].dtype) * 128

    raw_pix = np.array([], dtype=video["Y"].dtype)
    for f in range(len(video["Y"])):
        for c in ["Y", "U", "V"]:
            raw_pix = np.concatenate([raw_pix, np.ndarray.flatten(video[c][f])])

    if automatic_file_name_extension:
        video_width = video["Y"].shape[2]
        video_height = video["Y"].shape[1]
        bitdepth = 8 if video["Y"].dtype == np.uint8 else 10
        subsampling_scheme = "4"
        subsampling_scheme += str(4 * video["U"].shape[2] // video_width) 
        subsampling_scheme += str(subsampling_scheme[-1] if video_height == video["U"].shape[1] else 0)

        filename_extension = "_" + str(video_width) + "x" + str(video_height) + "_" + str(bitdepth) + "bit_" + subsampling_scheme
        path = path.replace(".yuv", filename_extension + ".yuv")

    np.ndarray.tofile(raw_pix, sep="", file=path)

    return path


def __get_bandwidth_factor__(subsampling_scheme):
    '''
        Computes the relative bandwidth required for the video compared to a video in 4:4:4 format

        Parameters:
            subsampling_scheme (String): Subsampling used for the chroma channels of the video. Common options are: 444, 422, 420

        Returns:
            bandwidth_factor (float): Required bandwidth of the video compared to a YUV-Video stored in 4:4:4 format
    '''
    horizontal_sampling_reference = int(subsampling_scheme[0])

    subsampling_scheme_sum = 0

    for i in range(len(subsampling_scheme)):
        subsampling_scheme_sum += int(subsampling_scheme[i])

    return subsampling_scheme_sum / horizontal_sampling_reference / 3


def __get_subsampling_factors__(subsampling_scheme):
    '''
        Returns the horizontal and vertical subsampling factors for the given subsampling scheme

        Parameters:
            subsampling_scheme (String): Subsampling used for the chroma channels of the video. Common options are: 444, 422, 420

        Returns:
            subsampling_factors (tuple): Tuple of the horizontal and vertical subsampling factors (h_sub, v_sub)
    '''
    assert int(subsampling_scheme[1]) == int(subsampling_scheme[2]) or int(subsampling_scheme[2]) == 0

    horizontal_sampling_reference = int(subsampling_scheme[0])

    horizontal_subsampling_factor = horizontal_sampling_reference / int(subsampling_scheme[1])
    vertical_subsampling_factor = 1 if int(subsampling_scheme[1]) == int(subsampling_scheme[2]) else 2

    return (round(horizontal_subsampling_factor), vertical_subsampling_factor)