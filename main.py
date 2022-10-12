from time import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import click
import time

from skimage import transform
from skimage.color import rgb2gray, rgba2rgb
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.filters import threshold_otsu, threshold_local
from psd_tools import PSDImage 

class Img_Reg():

    def __init__(self, img_target_path, img_source_path, n, ratio = 0.8, eps = 0.8, all_plots = False, save_all=False, save_results=False, iter=False, recovery=False):
        """Initialises the Img_Reg class.

        Parameters
        ----------
        img_target_path : string
            The path where the annotated score can be located.
        img_source_path : string
            The path where the clean score can be located.
        n : int
            Records which image pair is being registered for saving.
        ratio : float
            The acceptance difference ratio for SIFT matches (default 0.8).
        eps : float
            The threshold for the RMSE of a transformation when iteration is being used (default 0.8).
        all_plots : bool, optional
            Determines whether plots are displayed, by default False
        save_all : bool, optional
            Determines whether plots are saved, by default False
        save_results : bool, optional
            Determine whether result metrics are saved, by default False
        iter : bool, optional
            Determines whether to iterate, by default False
        recovery : bool, optional
            Determines whether to calculate recovery metric, by default False
        """
        self.img_target, self.img_source = cv2.imread(img_target_path), cv2.imread(img_source_path)
        self.n = n
        self.ratio = ratio
        self.eps = eps
        self.L = 0
        self.all_plots = all_plots
        self.save_all = save_all
        self.save_results = save_results
        self.iter=iter
        self.recovery=recovery
        

    def _format_img(self, img):
        """Converts an rgba or rgb image to grayscale for use in _registered_rgb_plot.

        Parameters
        ----------
        img : numpy.ndarray
            The image to be formatted.

        Returns
        -------
        numpy.ndarray
            Grayscale image.
        """
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                img = rgba2rgb(img)
            if img.shape[2] == 3:
                img = rgb2gray(img)
        return img

    def _bgr2rgb(self, img):
        """Acts as a wrapper for the openCV function which converts an bgr image (used in openCV) to an rgb image (used in scikit-image).

        Parameters
        ----------
        img : numpy.ndarray
            The rbg image.

        Returns
        -------
        numpy.ndarray
            The rgb image.
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _sift_extract(self, img):
        """Extracts SIFT keypoints and descriptors from an image using openCV.

        Parameters
        ----------
        img : numpy.ndarray
            The image.

        Returns
        -------
        keypoints : numpy.ndarray
            A numpy array of all the keypoints.
        descriptors : numpy.ndarray
            A numpy array of all the image descriptors.
        
        """
        # Convert image to grayscale.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Extract key points and SIFT descriptors.
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)  
        # We must extract positions of keypoints.
        keypoints = np.array([p.pt for p in keypoints]).T  
        return keypoints, descriptors

    def _sift_match(self, descriptors_target, descriptors_source):
        """Uses a brute force matcher to find matching keypoints in the images based on their descriptors.

        Parameters
        ----------
        descriptors_target : numpy.ndarray
            The keypoint descriptors from the annotated image.
        descriptors_source : numpy.ndarray
            The keypoint descriptors from the clean image.

        Returns
        -------
        numpy.ndarray
            The pairs of indices of matching keypoints in each image.
        """
        # Match descriptor and obtain two best matches.
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors_target, descriptors_source, k=2)
        # Initialize output variable.
        fit_pos = np.array([], dtype=np.int32).reshape((0, 2))
        matches_num = len(matches)
        for i in range(matches_num):
            # Obtain the good match if the ratio is smaller than the ratio.
            if matches[i][0].distance <= self.ratio * matches[i][1].distance:
                temp = np.array([matches[i][0].queryIdx,
                                 matches[i][0].trainIdx])
                # Adds indices of good matches to the output variable.
                fit_pos = np.vstack((fit_pos, temp))
        return fit_pos

    def _estimate_affine(self, pts_s, pts_t):
        """Estimate the affine transformation matrix from the point correspondences.

        Parameters
        ----------
        pts_s : numpy.ndarray
            The selected keypoints in the clean image.
        pts_t : numpy.ndarray
            The corresponding selected keypoints in the annotated image.

        Returns
        -------
        numpy.ndarray
            The 2x3 affine transformation matrix.

        
        Method
        -------
            To estimate an affine transformation between two images,
            at least 3 corresponding points are needed.
            In this case, 6-parameter affine transformation are taken into
            consideration, which is shown as follows:
            
            | x' | = | a b | * | x | + | tx |\n
            | y' | \f\f  | c d | \f\f  | y |  \f\f | ty |

            For 3 corresponding points, 6 equations can be formed as below:
            
            | x1 y1 0  0  1 0 \f|\f\f\f\f   | a  |\f\f   | x1' |\n
            | 0  0  x1 y1 0 1 \f|\f\f\f\f   | b  |\f\f   | y1' |\n
            | x2 y2 0  0  1 0 | \a*\a       | c  |   =   | x2' |\n
            | 0  0  x2 y2 0 1 |\f\f\f\f     | d\a |\f\f  | y2' |\n
            | x3 y3 0  0  1 0 |\f\f\f\f     | tx |\f\f   | x3' |\n
            | 0  0  x3 y3 0 1 |\f\f\f\f     | ty |\f\f   | y3' |\n

            |----> M <----|            \f\f | theta | \f  | b |

            Solve the equation to compute theta by:  theta = M \ b
            Thus, affine transformation can be obtained as:

            A =        | a b |             \f\f t = | tx |
             \f\f\f\f  | c d |    \f\f\f\f\f\f\f     | ty |

            With the final transformation matrix given by:

            T =       | a b tx |
              \f\f\f  | c d ty |
        """
        # Get the number of corresponding points.
        pts_num = pts_s.shape[1]
        # Form the matrix M with 6 columns, since the affine transformation has 6 parameters here.
        M = np.zeros((2 * pts_num, 6))
        for i in range(pts_num):
            temp = [[pts_s[0, i], pts_s[1, i], 0, 0, 1, 0],
                    [0, 0, pts_s[0, i], pts_s[1, i], 0, 1]]
            M[2 * i: 2 * i + 2, :] = np.array(temp, dtype=object)
        # Form the matrix b containing the target points.
        b = pts_t.T.reshape((2 * pts_num, 1))
        try:
            # Solve the linear equation using the least squares method, and form the matrixes.
            theta = np.linalg.lstsq(M, b, rcond=None)[0]
            A = theta[:4].reshape((2, 2))
            t = theta[4:]
        except np.linalg.linalg.LinAlgError:
            # If M is singular matrix, return None.
            A = None
            t = None
        T = np.hstack((A, t))
        return T

    def _sift_plot(self, keypoints_target, keypoints_source, matches):
        """Generates plot showing SIFT keypoints and the corresponding matches in each image.

        Parameters
        ----------
        keypoints_target : numpy.ndarray
            Keypoints in the annotated image.
        keypoints_source : numpy.ndarray
            Keypoints in the clean image
        matches : numpy.ndarray
            Indices of the SIFT selected matching keypoints from each image.
        """
        # Convert the keypoints to correct the form by transposing and swap columns (Use a copy so it doesn't effect the result).
        keypoints_target = keypoints_target.copy().T
        keypoints_target[:, [0,1]] = keypoints_target[:,[1,0]]
        keypoints_source = keypoints_source.copy().T
        keypoints_source[:, [0,1]] = keypoints_source[:,[1,0]]
        # Plot the keypoints and matches.
        fig, ax = plt.subplots(1, 1, figsize=(11, 8))
        plot_matches(ax, self._bgr2rgb(self.img_target), self._bgr2rgb(self.img_source),
                     keypoints_target, keypoints_source, matches)
        ax.axis('off')
        ax.set_title("Target Image vs. Source Image\n"
                        "(all SIFT keypoints and matches)")
        plt.tight_layout()
        if self.save_all:
            plt.savefig(f"Results/img{self.n:02d}_sift_points_matches.png")
        if self.all_plots:
            plt.text(-0.15, -20, f"img{self.n:02d}, plot 1/7", ha='left', fontsize=9)
            plt.show()
        plt.close()
        # Don't display plot of keypoints, just save it.
        if self.save_all:
            fig, ax = plt.subplots(1, 1, figsize=(11, 8))
            plot_matches(ax, self._bgr2rgb(self.img_target), self._bgr2rgb(self.img_source),
                        keypoints_target, keypoints_source, np.array([]))
            ax.axis('off')
            ax.set_title("Target Image vs. Source Image\n"
                            "(all SIFT keypoints)")
            plt.tight_layout()
            plt.savefig(f"Results/img{self.n:02d}_sift_points.png", dpi = 1000, bbox_inches = "tight")
            plt.close()
        return

    def _sift_ransac_plot(self, keypoints_target, keypoints_source, matches, matches_ransac):
        """Generates a plot comparing the SIFT and RANSAC selected matches in the images.

        Parameters
        ----------
        keypoints_target : numpy.ndarray
            Keypoints in the annotated image.
        keypoints_source : numpy.ndarray
            Keypoints in the clean image
        matches : numpy.ndarray
            Indices of the SIFT selected matching keypoints from each image.
        matches_ransac : numpy.ndarray
            Indices of the Ransac selected matching keypoints from each image.
        """
        # Convert the keypoints to correct the form by transposing and swap columns (Use a copy so it doesn't effect the result).
        keypoints_target = keypoints_target.copy().T
        keypoints_target[:, [0,1]] = keypoints_target[:,[1,0]]
        keypoints_source = keypoints_source.copy().T
        keypoints_source[:, [0,1]] = keypoints_source[:,[1,0]]
        # Plot and save the SIFT and RANSAC matches.
        if self.save_all:
                # SIFT matches plot
                fig, ax = plt.subplots(1, 1, figsize=(11, 8))
                plot_matches(ax, self._bgr2rgb(self.img_target), self._bgr2rgb(self.img_source),
                        keypoints_target, keypoints_source, matches, only_matches=True)
                ax.axis('off')
                ax.set_title("Target Image vs. Source Image\n"
                                    f"({len(matches)} SIFT matches)")
                plt.tight_layout()
                plt.savefig(f"Results/img{self.n:02d}_sift_matches.png", dpi = 1000, bbox_inches = "tight")
                if self.all_plots:
                    plt.text(-0.15, -20, f"img{self.n:02d}, plot 2/7", ha='left', fontsize=9)
                    plt.show()
                plt.close()
                # RANSAC matches plot
                fig, ax = plt.subplots(1, 1, figsize=(11, 8))
                plot_matches(ax, self._bgr2rgb(self.img_target), self._bgr2rgb(self.img_source), keypoints_target,
                                keypoints_source, matches_ransac, only_matches=True)
                ax.axis('off')
                ax.set_title("Target Image vs. Source Image\n"
                                    f"({len(matches_ransac)} RANSAC matches)")
                plt.tight_layout()
                plt.savefig(f"Results/img{self.n:02d}_ransac_matches.png", dpi = 1000, bbox_inches = "tight")
                if self.all_plots:
                    plt.text(-0.15, -20, f"img{self.n:02d}, plot 3/7", ha='left', fontsize=9)
                    plt.show()
                plt.close()
        # Just plot the SIFT and RANSAC matches together if not saving.
        elif self.all_plots:
            fig, ax = plt.subplots(2, 1, figsize=(11, 8))
            plot_matches(ax[0], self._bgr2rgb(self.img_target), self._bgr2rgb(self.img_source),
                            keypoints_target, keypoints_source, matches, only_matches=True)
            ax[0].axis('off')
            ax[0].set_title("Target Image vs. Source Image\n"
                                f"({len(matches)} SIFT matches)")

            plot_matches(ax[1], self._bgr2rgb(self.img_target), self._bgr2rgb(self.img_source), keypoints_target,
                            keypoints_source, matches_ransac, only_matches=True)
            ax[1].axis('off')
            ax[1].set_title("Target Image vs. Source Image\n"
                                f"({len(matches_ransac)} RANSAC matches)")
            plt.tight_layout()
            plt.text(-0.15, 1.1, f"img{self.n:02d}, plot 2/7", fontsize=9, transform=ax[0].transAxes)
            plt.text(-0.15, 1.1, f"img{self.n:02d}, plot 3/7", fontsize=9, transform=ax[1].transAxes)
            plt.show()
        return

    def _registered_rgb_plot(self, img_post):
        """Generates an rgb plot comparing the annotated image (red channel) to the registered clean image (blue and green channels).

        Parameters
        ----------
        img_post : numpy.ndarray
            The registered clean image.
        """
        image0, image1 = img_post, self.img_target
        # Convert the images to grayscale.
        image0, image1 = self._format_img(image0), self._format_img(image1)
        nr, nc = image0.shape
        # build an RGB image of the registered sequence.
        reg_im = np.zeros((nr, nc, 3))
        reg_im[..., 0] = image1
        reg_im[..., 1] = image0
        reg_im[..., 2] = image0
        # build an RGB image of the registered image.
        target_im = np.zeros((nr, nc, 3))
        target_im[..., 0] = image0
        target_im[..., 1] = image0
        target_im[..., 2] = image0
        # Show the result
        fig, axes = plt.subplots(1, 2, figsize=(11, 8))
        ax = axes.ravel()
        ax[0].imshow(reg_im)
        ax[0].set_title("Registered Sequence")
        ax[1].imshow(target_im)
        ax[1].set_title("Target")
        fig.tight_layout()
        if self.save_all:
            plt.savefig(f"Results/img{self.n:02d}_rgb.png", dpi = 1000, bbox_inches = "tight")
        if self.all_plots:
            plt.text(-2200, -100, f"img{self.n:02d}, plot 5/7", fontsize=9)
            plt.show()
        plt.close()
        return

    def _registered_sequence_plot(self, img_post):
        """Generates plot of the target, registered, and source images adjacent for easy comparison.

        Parameters
        ----------
        img_post : numpy.ndarray
            The registered clean image.
        """
        fig, axes = plt.subplots(1, 3, figsize=(11, 8))
        ax = axes.ravel()
        ax[0].imshow(self._bgr2rgb(self.img_target))
        ax[0].set_title('Target Image')
        ax[1].imshow(self._bgr2rgb(img_post))
        ax[1].set_title('Registered Image')
        ax[2].imshow(self._bgr2rgb(self.img_source))
        ax[2].set_title('Source Image')
        fig.tight_layout()
        if self.save_all:
            plt.savefig(f"Results/img{self.n:02d}_registered_sequence.png", dpi = 1000, bbox_inches = "tight")
        if self.all_plots:
            plt.text(-8000, -150, f"img{self.n:02d}, plot 4/7", fontsize=9)
            plt.show()
        plt.close()
        return

    def _registered_diff_plot(self, img_post):
        """Generates a plot with two images. The first is the difference you get when subtracting the clean/source image from the annotated/target image.
        The second is like the first but where both images are binarised/thresholded prior to subtraction using Otsu thresholding.

        Parameters
        ----------
        img_post : numpy.ndarray
            The registered clean iamge.
        plots : boolean
            Determines whether the plot will be shown or not.

        Returns
        -------
        numpy.ndarray
            The difference image for saving.
        """
        # Convert both images to grayscale.
        target_gray = cv2.cvtColor(self.img_target, cv2.COLOR_BGR2GRAY)
        post_gray = cv2.cvtColor(img_post, cv2.COLOR_BGR2GRAY)
        # Calculate thresholds and obtain thresholded images.
        thresh1 =  threshold_otsu(target_gray)
        thresh2 =  threshold_otsu(post_gray)
        _, target_bin = cv2.threshold(target_gray, thresh1, 255, cv2.THRESH_BINARY)
        _, post_bin = cv2.threshold(post_gray, thresh2, 255, cv2.THRESH_BINARY)
        # Obtain the difference images.
        img_diff = 255-cv2.subtract(img_post, self.img_target) 
        img_diff_bin = 255-cv2.subtract(post_bin, target_bin) 
        # Plot or save to folder if desired and not iterative.
        if self.iter == False:
            fig, ax = plt.subplots(1, 1, figsize=(11, 8))
            # Plot and/or save difference image.
            ax.imshow(self._bgr2rgb(img_diff))
            ax.set_title('Difference Image')
            fig.tight_layout()
            # Always save the difference image.
            plt.savefig(f"Results/img{self.n:02d}_diff.png", dpi = 1000, bbox_inches = "tight")
            if self.all_plots:
                plt.text(-0.15, -20, f"img{self.n:02d}, plot 6/7", ha='left', fontsize=9)
                plt.show()
            plt.close()
            # Plot and/or save binarised difference image.
            fig, ax = plt.subplots(1, 1, figsize=(11, 8))
            ax.imshow(self._bgr2rgb(img_diff_bin))
            ax.set_title('Binarised Difference Image')
            fig.tight_layout()
            if self.save_all:
                plt.savefig(f"Results/img{self.n:02d}_diff_bin.png", dpi = 1000, bbox_inches = "tight")
            if self.all_plots:
                plt.text(-0.15, -20, f"img{self.n:02d}, plot 7/7", ha='left', fontsize=9)
                plt.show()
            plt.close()
        # Note that the return is only used in iterative case.
        return img_diff

    def _save_to_file_iterative(self, img_diff, img_post):
        """Save the difference and registered image to the Results Iterative folder.

        Parameters
        ----------
        img_diff : numpy.ndarray
            The difference image between the target and reference image.
        """
        # Create a folder if one doesn't already exist.
        os.makedirs("Results Iterative", exist_ok=True)
        # Save to the folder, naming the file with the correct number.
        cv2.imwrite(f"Results Iterative/img{self.n:02d}_diff.png", img_diff)
        cv2.imwrite(f"Results Iterative/img{self.n:02d}_registered.png", img_post)
        return

    def _iterate(self, kp_t, kp_s, M):
        """Determine whether another iteration is required based on the RMSE.

        Parameters
        ----------
        kp_t : numpy.ndarray
            The selected keypoints from the target image.
        kp_s : numpy.ndarray
            The selected keypoints from the source image.
        M : numpy.ndarray
            The affine transformation matrix.

        Returns
        -------
        boolean
            To iterate or not to iterate?
        """
        rmse = self._rmse(kp_t, kp_s, M)
        click.echo(f"Iteration: {self.L} \nRMSE: {rmse}")
        # Compare the RMSE to the threshold to determine whether to iterate.
        if rmse < self.eps:
            print("Done.\n")
            return False
        else:
            self.L += 1
            return True 
    
    def _rmse(self, kp_t, kp_s, M):
        """Calculate the root-mean-square error of the transformation.

        Parameters
        ----------
        kp_t : numpy.ndarray
            The selected keypoints from the target image.
        kp_s : numpy.ndarray
            The selected keypoints from the source image.
        M : numpy.ndarray
            The affine transformation matrix.

        Returns
        -------
        float
            The RMSE.
        """
        # Convert the keypoints to correct the form by transposing and swap columns (Use a copy so it doesn't effect the result).
        kp_t = kp_t.copy().T
        kp_t[:, [0,1]] = kp_t[:,[1,0]]
        kp_s = kp_s.copy().T
        kp_s[:, [0,1]] = kp_s[:,[1,0]]
        # Find the coordinates of the transformed points in the registered image.
        kp_s_dash = np.empty((0,2))
        for point in kp_s:
            old = np.array([[point[0]], [point[1]], [1]])
            new = M @ old
            kp_s_dash = np.vstack([kp_s_dash, np.array(new[0:2, 0])])
        # Calculate and display the root mean squared error (RMSE).
        diff = np.linalg.norm(kp_s_dash-kp_t, axis=1)
        rmse = np.sqrt(sum(diff)/diff.shape[0])
        return rmse
    
    def _compute_xcorr(self, marked_r, clean_r):
        """Compute the cross-correlation coefficient between the annotated and clean registered score.

        Parameters
        ----------
        marked_r : numpy.ndarray
            The annotated score image.
        clean_r : numpy.ndarray
            The registered clean score image.

        Returns
        -------
        float
            The cross-correlation coefficient.
        """
        marked_values = rgb2gray(marked_r).ravel()
        marked_values = marked_values / np.linalg.norm(marked_values)
        clean_values = rgb2gray(clean_r).ravel()
        clean_values = clean_values / np.linalg.norm(clean_values)
        return np.correlate(marked_values, clean_values)[0]

    def _otsu(self, a):
        """Returns True/False array for which pixels in an image are above the otsu threshold.

        Parameters
        ----------
        a : numpy.ndarray
            The image being checked.

        Returns
        -------
        numpy.ndarray
            True/False values for whether pixels are above or below the threshold.
        """
        gray = rgb2gray(a)
        thresh = threshold_otsu(gray)
        return gray > thresh

    def _local(self, a, bs=257):
        """Returns True/False array for which pixels in an image are above the local threshold.

        Parameters
        ----------
        a : numpy.ndarray
            The image being checked.
        bs : int
            The block size for local thresholding.

        Returns
        -------
        numpy.ndarray
            True/False values for whether pixels are above or below the threshold.
        """
        gray = rgb2gray(a)
        thresh = threshold_local(gray, block_size=bs)
        return gray > thresh

    def _compute_recovery_score(self, marked_r, clean_r, local_bs=513):
        """Function to compute the recovery metric using the ground truth psd files.

        Parameters
        ----------
        marked_r : numpy.ndarray
            The annotated image.
        clean_r : numpy.ndarray
            The clean registered image.
        local_bs : int, optional
            blocksize for local thresholding, by default 513

        Returns
        -------
        float
            The recovery score.
        """
        annot = PSDImage.open(f"psd/img{self.n:02d}.psd")[1].numpy()
        if annot.shape[2] == 4:
                annot = rgba2rgb(annot)
        # Finding the difference image (in terms of T/F values in regard to the threshold)
        # If pixels are above the threshold in both the marked and clean registered image (i.e. the printed score) they are set to False.
        annot_pred = self._local(marked_r, bs=local_bs) ^ self._otsu(clean_r) 
        # Any pixels in the ground truth that are also above the threshold are found in T/F array.
        annot_gt = self._otsu(annot)
        try:
            # Find the number of pixels which are above the threshold in both the ground truth and difference image.
            recovery = sum((annot_gt & annot_pred).ravel())
        except ValueError:
            recovery = 0
        target = sum(annot_gt.ravel())
        # Return what fraction of the annotations in the GT are recovered in the difference image.
        return recovery / target

    def register(self):
        """Main method function which performs the image registration pipeline. 

        Parameters
        ----------
        plots : bool, optional
            Determines whether to display key plots (comparison and difference plots), by default True
        all_plots : bool, optional
            Determines whether to display all plots (SIFT, RANSAC, and rgb plot), by default False
        iter : bool, optional
            Determines whether to use iteration, by default False

        Method
        ----------
        1. SIFT keypoints are extracted from the images and brute force matched using their descriptors.
        2. RANSAC is used to filter out false matches.
        3. The affine transformation matrix is calculated from the correct matches using a least squares method.
        4. The source image is warped using this matrix to obtain the registered image.
        5. A number of plots are displayed and the difference image between the target and registered image is saved.
        """
        # Extract SIFT keypoints and descriptors from source image and target images.
        keypoints_target, descriptors_target = self._sift_extract(self.img_target)
        keypoints_source, descriptors_source = self._sift_extract(self.img_source)
        # Find the indices of matching keypoints using keypoint descriptors, then extract these keypoints for RANSAC.
        matches = self._sift_match(descriptors_target, descriptors_source)
        keypoints_target2 = keypoints_target[:, matches[:, 0]]
        keypoints_source2 = keypoints_source[:, matches[:, 1]]
        # Apply RANSAC to find inlying keypoints.
        _, inliers = ransac((keypoints_target2.T, keypoints_source2.T), transform.AffineTransform, min_samples=3,
                                        residual_threshold=1, max_trials=2000) 
        outliers = inliers == False
        inliers = np.where(inliers == True)[0]
        # Obtain indices of corresponding keypoints in RANSAC matches, and then extract these keypoints.
        matches_ransac = matches[inliers]
        keypoints_target3 = keypoints_target2[:, inliers]
        keypoints_source3 = keypoints_source2[:, inliers]
        # Use the RANSAC selected keypoints to estimate the affine transformation matrix.
        M = self._estimate_affine(keypoints_source3, keypoints_target3)
        # Warp moving image obtaining the registered image.
        rows, cols, _ = self.img_target.shape
        img_post = cv2.warpAffine(self.img_source, M, (cols, rows), borderValue=(255,255,255))
        # Save the registered image unless iterative then it gets saved later on.
        if not self.iter:
            os.makedirs("Results", exist_ok=True)
            cv2.imwrite(f"Results/img{self.n:02d}_registered.png", img_post)
        # Save the result data for the registered image and the annotated target.
        if self.save_results:
            if self.iter:
                rmse = self._rmse(keypoints_target3, keypoints_source3, M)
                with open('Results/Results.txt', 'a') as f:
                    if self.recovery:
                        f.write(f"[{self.n}, {self.L}, {rmse:.5f}, {self._compute_xcorr(self.img_target, img_post):.5f}, {self._compute_recovery_score(self.img_target, img_post):.5f}, {len(keypoints_source2[0])}, {len(keypoints_source3[0])}],\n")
                    else:
                        f.write(f"[{self.n}, {self.L}, {rmse:.5f}, {self._compute_xcorr(self.img_target, img_post):.5f}, {len(keypoints_source2[0])}, {len(keypoints_source3[0])}],\n")
            else:
                with open('Results/Results.txt', 'a') as f:
                    if self.recovery:
                        f.write(f"[{self.n}, {self._rmse(keypoints_target3, keypoints_source3, M):.5f}, {self._compute_xcorr(self.img_target, img_post):.5f}, {self._compute_recovery_score(self.img_target, img_post):.5f}, {len(keypoints_source2[0])}, {len(keypoints_source3[0])}],\n")
                    else:
                        f.write(f"[{self.n}, {self._rmse(keypoints_target3, keypoints_source3, M):.5f}, {self._compute_xcorr(self.img_target, img_post):.5f}, {len(keypoints_source2[0])}, {len(keypoints_source3[0])}],\n")
        # Call functions to display and/or save the results plots.
        if (self.all_plots or self.save_all) and (self.iter==False):
            self._sift_plot(keypoints_target, keypoints_source, matches)
            self._sift_ransac_plot(keypoints_target, keypoints_source, matches, matches_ransac)
            self._registered_sequence_plot(img_post)
            self._registered_rgb_plot(img_post)
            img_diff = self._registered_diff_plot(img_post)
        elif self.iter==False:
            # So that we are always saving the difference image.
            img_diff = self._registered_diff_plot(img_post)
        # Perform iteration using RMSE:
        if self.iter == True:
            img_diff = self._registered_diff_plot(img_post)
            if self._iterate(keypoints_target3, keypoints_source3, M) == True:
                    # Update the target image to the registered image.
                    self.img_source = img_post
                    self.register()
            else:
                self._save_to_file_iterative(img_diff, img_post)
        return 
        
if __name__ == "__main__":
    """ Please note the following to prevent confusion:
    target = annotated (= reference).
    source = clean (= moving).
    We are registering the moving image to the reference image.
    """
    # function to obtain the image paths from a folder.
    def load_image_paths_from_folder(folder):
        """Obtains the paths of all the images in a folder 

        Parameters
        ----------
        folder : string
            The name of the folder where the images are.

        Returns
        -------
        list
            Returns a sorted list containing the paths of all the images from the folder.
        """
        images = []
        for filename in os.listdir(folder):
            # This is to ignore the metadata files that appear when reading from a folder on MacOS:
            if filename != ".DS_Store":
                img = os.path.join(folder,filename)
                if img is not None:
                    images.append(img)
        return sorted(images)
    # Initialise click, set up, and call main function.
    @click.command()
    @click.option('--all_plots', prompt="Do you want plots to be displayed (Y/N)?", help='Do you want all plots to be displayed.')
    @click.option('--save_all', prompt="Do you want all plots to be saved (The registered image and difference plot are saved by default) (Y/N)?", help='Do you want all images to be saved. Only registered image and difference plot are saved by default.')
    @click.option('--save_results', prompt="Do you want result metrics to be saved to a text file (Y/N)?", help='Do you want to save result data to a text file.')
    @click.option('--iter', default='N', help='Do you want to iteratively register till RMSE < 0.8 (Y/N)?')
    @click.option('--recovery', default='N', help='Do you want to calculate the recovery of annotations using ground truth images (Y/N)?')
    def master(all_plots, save_all, save_results, iter, recovery):
        start = time.time()
        click.echo("Starting...")
        # Convert inputs from Y/N to True/False.
        if all_plots == 'N':
            all_plots = False
        else:
            all_plots = True
        if save_all == 'N':
            save_all = False
        else:
            save_all = True
        if save_results == 'N':
            save_results = False
        else:
            save_results = True
        if iter == 'N':
            iter = False
        else:
            iter = True
        if recovery == 'N':
            recovery = False
        else:
            recovery = True
        # Force all_plots and save_all to False if using iteration.
        if iter:
            all_plots=False
            save_all=False
        # Create Results folder:
        os.makedirs("Results", exist_ok=True)
        # Create results file.
        if save_results:
            if iter:
                with open('Results/Results.txt', 'w') as f:
                    if recovery:
                        f.write("[Image Pair, Iteration, RMSE, xcorr, Recovery, SIFT matches, RANSAC matches],\n")
                    else:
                        f.write("[Image Pair, Iteration, RMSE, xcorr, SIFT matches, RANSAC matches],\n")
            else:
                with open('Results/Results.txt', 'w') as f:
                    if recovery:
                        f.write("[Image Pair, RMSE, xcorr, Recovery, SIFT matches, RANSAC matches],\n")
                    else:
                        f.write("[Image Pair, RMSE, xcorr, SIFT matches, RANSAC matches],\n")
        # Obtain the paths of the annotated and clean images.
        target_img_paths = load_image_paths_from_folder("Annotated Images") 
        source_img_paths = load_image_paths_from_folder("Clean Images") 
        # Perform the registration on each pair of images, saving the difference images between the target and registered images.
        for i in range(len(target_img_paths)):
            if iter:
                click.echo(f"Image pair {i+1}")
            else:
                click.echo(f"{((i+1)/len(target_img_paths)):.0%} complete...")
            s = Img_Reg(target_img_paths[i], source_img_paths[i], i+1, all_plots = all_plots, save_all=save_all, save_results=save_results, iter=iter, recovery=recovery)
            s.register() 
        end = time.time()
        click.echo(f"Done, time taken: {end-start:.5f}s")
    master()