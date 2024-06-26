import numpy as np
import cv2
import math
import random


def RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement):
    """
    This function takes in `matched_pairs`, a list of matches in indices
    and return a subset of the pairs using RANSAC.
    Inputs:
        matched_pairs: a list of tuples [(i, j)],
            indicating keypoints1[i] is matched
            with keypoints2[j]
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        *_agreement: thresholds for defining inliers, floats
    Output:
        largest_set: the largest consensus set in [(i, j)] format

    HINTS: the "*_agreement" definitions are well-explained
           in the assignment instructions.
    """
    assert isinstance(matched_pairs, list)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    ## START


    ## END
    largest_set = []
    max_inliers = 1
    
    for _ in range(10):  # Repeat the random selection 10 times
        inliers_set = []
        random_pair = random.choice(matched_pairs)
        inliers_set.append(random_pair)  # Ensure the initial match is included in the consensus set

        for pair in matched_pairs:
            if pair == random_pair:  # Skip the initially randomly selected pair
                continue

            kp1 = keypoints1[pair[0]]
            kp2 = keypoints2[pair[1]]

            ori_diff = abs(kp1[3] - kp2[3])
            if ori_diff > np.pi:
                ori_diff = 2 * np.pi - ori_diff

            if math.degrees(ori_diff) <= orient_agreement:
                scale_diff = abs(kp2[2] / kp1[2] - 1)
                if scale_diff <= scale_agreement:
                    inliers_set.append(pair)

        if len(inliers_set) > max_inliers:
            max_inliers = len(inliers_set)
            largest_set = inliers_set

    return largest_set


def FindBestMatches(descriptors1, descriptors2, threshold):
    """
    This function takes in descriptors of image 1 and image 2,
    and find matches between them. See assignment instructions for details.
    Inputs:
        descriptors: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
    Outputs:
        matched_pairs: a list in the form [(i, j)] where i and j means
                       descriptors1[i] is matched with descriptors2[j].
    """
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    ## START
    ## the following is just a placeholder to show you the output format
    y1 = descriptors1.shape[0]
    y2 = descriptors2.shape[0]
    matched_pairs = []

    for i in range(y1):
        temp = np.zeros(y2)
        for j in range(y2):
            temp[j] = math.acos(np.dot(descriptors1[i], descriptors2[j]))

        compare = sorted(range(len(temp)), key=lambda k: temp[k])
        if temp[compare[0]] / temp[compare[1]] < threshold:
            matched_pairs.append((i, compare[0]))


    ## END
    return matched_pairs


def KeypointProjection(xy_points, h):
    """
    This function projects a list of points in the source image to the
    reference image using a homography matrix `h`.
    Inputs:
        xy_points: numpy array, (num_points, 2)
        h: numpy array, (3, 3), the homography matrix
    Output:
        xy_points_out: numpy array, (num_points, 2), input points in
        the reference frame.
    """
    assert isinstance(xy_points, np.ndarray)
    assert isinstance(h, np.ndarray)
    assert xy_points.shape[1] == 2
    assert h.shape == (3, 3)

    # START
    # Convert xy_points to homogeneous coordinates
    xy_points_homogeneous = np.hstack((xy_points, np.ones((xy_points.shape[0], 1))))

    # Perform projection (matrix multiplication, h*homogeneous(xy_points))
    xy_prj_homogeneous = np.dot(h, xy_points_homogeneous.T).T

    # Convert xy_prj to regular coordinates
    xy_points_out = xy_prj_homogeneous[:, :2] / xy_prj_homogeneous[:, 2:]

    # Replace zero values with a very large number to avoid division by zero
    xy_points_out[xy_points_out == 0] = 1e10


    # END
    return xy_points_out

def RANSACHomography(xy_src, xy_ref, num_iter, tol):
    """
    Given matches of keyponit xy coordinates, perform RANSAC to obtain
    the homography matrix. At each iteration, this function randomly
    choose 4 matches from xy_src and xy_ref.  Compute the homography matrix
    using the 4 matches.  Project all source "xy_src" keypoints to the
    reference image.  Check how many projected keyponits are within a `tol`
    radius to the coresponding xy_ref points (a.k.a. inliers).  During the
    iterations, you should keep track of the iteration that yields the largest
    inlier set. After the iterations, you should use the biggest inlier set to
    compute the final homography matrix.
    Inputs:
        xy_src: a numpy array of xy coordinates, (num_matches, 2)
        xy_ref: a numpy array of xy coordinates, (num_matches, 2)
        num_iter: number of RANSAC iterations.
        tol: float
    Outputs:
        h: The final homography matrix.
    """
    assert isinstance(xy_src, np.ndarray)
    assert isinstance(xy_ref, np.ndarray)
    assert xy_src.shape == xy_ref.shape
    assert xy_src.shape[1] == 2
    assert isinstance(num_iter, int)
    assert isinstance(tol, (int, float))
    tol = tol*1.0

    # START
    max_inliers = 0
    h = None
    N = xy_src.shape[0]

    for i in range(num_iter):
        sample_indices = np.random.choice(N, 4, replace=False)
        src_sample = xy_src[sample_indices]
        ref_sample = xy_ref[sample_indices]

        # Calculate the homography matrix using the selected samples
        H, _ = cv2.findHomography(src_sample, ref_sample)

        # Project source keypoints to the reference image
        xy_src_projected = KeypointProjection(xy_src, H)

        # Compute distances between projected keypoints and corresponding reference keypoints
        res_dis = np.sqrt(np.sum((xy_ref - xy_src_projected) ** 2, axis=1))

        # Count the number of inliers
        inliers = np.sum(res_dis < tol)

        # Update the homography matrix if the current iteration yields more inliers
        if inliers > max_inliers:
            h = H
            max_inliers = inliers

    return h


def FindBestMatchesRANSAC(
        keypoints1, keypoints2,
        descriptors1, descriptors2, threshold,
        orient_agreement, scale_agreement):
    """
    Note: you do not need to change this function.
    However, we recommend you to study this function carefully
    to understand how each component interacts with each other.

    This function find the best matches between two images using RANSAC.
    Inputs:
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        descriptors1, 2: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
        orient_agreement: in degrees, say 30 degrees.
        scale_agreement: in floating points, say 0.5
    Outputs:
        matched_pairs_ransac: a list in the form [(i, j)] where i and j means
        descriptors1[i] is matched with descriptors2[j].
    Detailed instructions are on the assignment website
    """
    orient_agreement = float(orient_agreement)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    matched_pairs = FindBestMatches(
        descriptors1, descriptors2, threshold)
    matched_pairs_ransac = RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement)
    return matched_pairs_ransac
