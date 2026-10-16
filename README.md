# Automatic_Panorama_Stitching
Automatically combines a sequence of overlapping photographs into a panorama.

--Running The System--

Required Libraries:

-argparse
-numpy
-cv2
-__future__/print_function
-PIL/Image, ImageDraw

How To Run The Program: The program runs through a parser in the terminal

In the terminal start with: python3 panorama_stitch.py

You will then input these arguments:

images: Input the name and file type of the images you would like to be stitched together

--dir: Select the name of a folder containing images that you would like to create a panorama of. Reads images in alphabetical order.

--ratio: Allows for changing of the ratio threshold of d1/d2 < r. Defaults to r=0.75

--iters: Allows for changing of the number of iterations used in RANSAC. Defaults to 2000

--threshold: Allows for changing of the inlier threshold. Defaults to 5

--model: Allows for the choice in what type of transformation is used. Options: affine, homography, both. Defaults to both.

--ref: Selects the reference image for a multi-image panorama. Can select image starting form index 0 where --ref 0 is the first/left-most image. Defaults to the middle image or teh second image if only 2.

--max_dim: Allows for the size of the resulting panorama image pixel size to be changed. Defaults to 1600

--output: Sets filename prefix for the image files. Defaults to panorama.

--seed: Selects the seeds in which RANSAC chooses its random samples. Defaults to 0.

Expected Input/Output:

Example Parser Input:

No directory:

$ python3 panorama_stitch.py dvd_left.png dvd_right.png --ratio 0.75 --iters 2000 --threshold 5 --model homography --ref 0 --max_dim 1600 --output panorama --seed 0

With a directory:

$ python3 panorama_stitch.py --dir dvd --ratio 0.75 --iters 2000 --threshold 5 --model homography --ref 0 --max_dim 1600 --output panorama --seed 0

Output: The output consists of pictures and a terminal textual output

 1. Pictures
    The program, depending on the options and number of images, produces at least two output images, one showing the matching features, color coded by inliers(green) and outliers(red), and the final resulting panorama. If you choose to do both an affine and homography, you will get images showing both the affine and homography matches, and both of their final panoramas. In a multi-image panorama, you will receive a matches image for each pair of images. 
    For example, with 3 images there will be a matches_(model)_0_1.png and matches_(model)_1_2.png showing the matches of the first and second image and second and third image.

 2. Terminal Text Output:
    The terminal will output items in this order:
    The images loaded and their size
    The reference image used for the transformation
    The number of keypoints found in each image

    It will then start with the homography stats if selected

    The pair of images and the number of matches between them, as well as the ratio threshold
    The number of inliers out of the matches
    The homography matrix H
    ^Repeat for each image pair^
    The raw output image size
    The actual output image size
    The size of the region in which the edges were blended
    The file name the panorama was saved into

    It will then show the affine stats if selected

    The pair of images and the number of matches between them, as well as the ratio threshold
    The number of inliers out of the matches
    The affine matrix H
    ^Repeat for each image pair^
    The raw output image size
    The actual output image size
    The size of the region in which the edges were blended for each image
    The file name the panorama was saved into

How To Reproduce My Main Experiments:


--Feature Matching Evaluation--

Image Pairing: dvd_left.png and dvd_right.png

r = 0.60 |
Matches: 85 |
Inliers: 54 |
% Retained: 63.53

<img src="results/matches_homography_60.png">

r = 0.75 |
Matches: 219 |
Inliers: 70 |
% Retained: 31.96

<img src="results/matches_homography_75.png">

r = 0.90 |
Matches: 489 |
Inliers: 103 |
% Retained: 21.06

<img src="results/matches_homography_90.png">

--RANSAC Evaluation--



--Affine versus Homography--

Image Pairing: uttower1.jpg and uttower2.jpg

Affine:

<img src="results/uttower_affine.png">

Homography:

<img src="results/uttower_homography.png">

--Multiple-Image Panorama--

Multiple Images: yosemite1.jpg, yosemite2.jpg, yosemite3.jpg, yosemite4.jpg

<img src="results/yosemite_homography.png">

The reference image used in this panorama is yosemite3.jpg


--AI Declaration--
