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

--threshold: Allows for changing of the inlier threshold. Defaults to 2.

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
    The reprojection error:
      Before RANSAC
      After RANSAC
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

Image Pairing: uttower1.jpg and uttower2.jpg

r = 0.60 |
Matches: 306 |
Inliers: 304 |
% Retained: 99.35

<img src="results/uttower_matches_homography_ratio60.png">

r = 0.75 |
Matches: 366 |
Inliers: 354 |
% Retained: 96.72

<img src="results/uttower_matches_homography_ratio75.png">

r = 0.90 |
Matches: 485 |
Inliers: 382 |
% Retained: 78.76

<img src="results/uttower_matches_homography_ratio90.png">


Between these two image pairs, the reasonable default for the ratio is about 0.75. If the ratio is too high, it creates too many matches and exceedingly sporadic matches that decreases the inlier retention rate significantly. At around 0.60, it could work but in certain image pairings this seems as if it would be too restrictive in the amount of matches it makes, as a majority of them remain as inliers. At a ratio threshold of 0.75, you get enough matches that there are some that are bad, but the retention rate is high enough that it is properly determining the correct homography. So to me it is adaptable enough to capture the widest range of different images.

--RANSAC Evaluation--

Inlier Threshold Test:

Image Pairing: uttower1.jpg and uttower2.jpg

threshold: 1 | matches: 366 | inliers: 307 | %: 83.88

Reprojection Error Medians:

Before RANSAC: 16.38 | After RANSAC inliers only: 0.39 | After RANSAC all: 0.46

<img src="results/uttower_matches_homography_thresh1.png">

threshold: 2 | matches: 366 | inliers: 351 | %: 95.90

Reprojection Error Medians:

Before RANSAC: 16.38 | After RANSAC inliers only: 0.50 | After RANSAC all: 0.53

<img src="results/uttower_matches_homography_thresh2.png">

threshold: 5 | matches: 366 | inliers: 354 | %: 96.72

Reprojection Error Medians:

Before RANSAC: 16.38 | After RANSAC inliers only: 0.46 | After RANSAC all: 0.49

<img src="results/uttower_matches_homography_thresh5.png">

At a threshold of 1, the number of inliers is significant, there still appears to be some close matches that are not counted as inliers. When we increase the threshold to 2, there is a significant jump to 351 inliers from 307 at threshold=1. If we try a higher threshold like 5, the inliers barely increase at all. So, for this homography the ideal threshold is at least 2.

Image Pairing: dvd_left.png and dvd_right.png

threshold: 2 | matches: 219 | inliers: 37 | %: 16.89

Reprojection Error Medians:

Before RANSAC: 36.77 | After RANSAC inliers only: 0.91 | After RANSAC all: 54.08

<img src="results/dvd_matches_homography_thresh2.png">

threshold: 5 | matches: 219 | inliers: 70 | %: 31.96

Reprojection Error Medians:

Before RANSAC: 36.77 | After RANSAC inliers only: 1.51 | After RANSAC all: 54.08

<img src="results/dvd_matches_homography_thresh5.png">

threshold: 26 | matches: 219 | inliers: 106 | %: 48.40 

Reprojection Error Medians:

Before RANSAC: 36.77 | After RANSAC inliers only: 3.06 | After RANSAC all: 62.26

<img src="results/dvd_matches_homography_thresh26.png">

threshold: 27 | matches: 219 | inliers: 107 | %: 48.86

Reprojection Error Medians:

Before RANSAC: 36.77 | After RANSAC inliers only: 9.44 | After RANSAC all: 28.46

<img src="results/dvd_matches_homography_thresh27.png">

For the dvd images at a threshold of 2, there are 37 inliers, but unlike the uttower images the number of inliers continues to rise at a high rate. The inlier increase from threshold 2 to 5 is 33 and the number increase between threshold 5 and 26 is 36. So the ideal threshold is slightly higher here around 5. 
The difference between the two images is that the dvds are taken at a much different perspective than the uttower images. It also has two distinct items that can create matches. Thresholds up to 26 focus solely on The Big Lebowski dvd cover, saying all the matches on the Ramones dvd are outliers, but at threshold 27 it then takes some matches on the ramones dvd as inliers, changing the homography for any larger thresholds. The projection error for inlier sharply increases from threshold 26 to 27, but then the overall error actually decreases sharply, so overall it is more accurate, but the inliers are less.

Homography Threshold = 26:

<img src="results/dvd_homography_thresh26.png">

Homography Threshold = 27:

<img src="results/dvd_homography_thresh27.png">

Because I wanted to take into account images like the dvds with larger changes in perspective, I set the default threshold to be 5 where most images would have the largest clear patterned increase in inliers.


RANSAC Iteration Tests:

Image Pairing: dvd_left.png and dvd_right.png

Iterations: 500 | matches: 219 | inliers: 35 | %:

Reprojection Error Medians:

Before RANSAC: 36.77 | After RANSAC inliers only: 2.63 | After RANSAC all: 60.27

<img src="results/dvd_matches_homography_iters500.png">

<img src="results/dvd_homography_iters500.png">

Iterations: 1000 | matches: 219 | inliers: 53 | %:

Reprojection Error Medians:

Before RANSAC: 36.77 | After RANSAC inliers only: 1.69 | After RANSAC all: 50.95

<img src="results/dvd_matches_homography_iters1000.png">

<img src="results/dvd_homography_iters1000.png">

Iterations: 1500 | matches: 219 | inliers: 70 | %:

Reprojection Error Medians:

Before RANSAC: 36.77 | After RANSAC inliers only: 1.51 | After RANSAC all: 54.08

<img src="results/dvd_matches_homography_iters1500.png">

<img src="results/dvd_homography_iters1500.png">

Iterations: 2000 | matches: 219 | inliers: 70 | %:

Reprojection Error Medians:

Before RANSAC: 36.77 | After RANSAC inliers only: 1.51 | After RANSAC all: 54.08

<img src="results/dvd_matches_homography_iters2000.png">

<img src="results/dvd_homography_iters2000.png">

After about 1500 iterations there was no further improvement within the model. From 1500 to 2000, the median errors all stayed the same as well as the number of inliers. The appearance of the homography improved as the number of iterations increased, as did the inlier error go down. I chose 2000 iterations as the default just in case as there may be certain image pairs that required more iterations, but it isn't so much more that it is extended the runtime of the program to an extreme amount.

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
