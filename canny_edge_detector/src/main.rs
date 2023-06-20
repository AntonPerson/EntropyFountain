use image::{DynamicImage, GrayAlphaImage, GrayImage, Luma, LumaA};
use ndarray::prelude::*;
use ndarray::{Array2, ArrayView2};
use std::f32;

/// # Canny edge detector
/// A Canny edge detector that can be used to detect edges in an image.
/// The detector is composed of the following steps:
/// 1. Gaussian blur
/// 2. Gradient calculation
/// 3. Non-maximum suppression
/// 4. Hysteresis thresholding
///
/// **References**
/// - Canny, J. (1986). A computational approach to edge detection. IEEE Transactions on pattern analysis and machine intelligence, 8(6), 679-698.
/// - https://en.wikipedia.org/wiki/Canny_edge_detector
/// - Canny Edge Detector: https://www.youtube.com/watch?v=sRFM5IEqR2w
/// - Sobel Operator: https://www.youtube.com/watch?v=uihBwtPIBxM
///
/// **Example**
/// ```
/// use canny_edge_detector::CannyEdgeDetector;
///
/// let input = image::open("input.png").unwrap();
/// let canny = CannyEdgeDetector::new(50.0, 100.0, 5, 1.0);
/// let edges = canny.process_image(&image);
/// edges.save("output.png").unwrap();
/// ```
pub struct CannyEdgeDetector {
    /// The low threshold for hysteresis thresholding.
    /// Must be less than the high threshold, and greater than 0.0.
    /// Used to determine weak edges.
    low_threshold: f32,
    /// The high threshold for hysteresis thresholding.
    /// Must be greater than the low threshold.
    /// Used to determine strong edges.
    high_threshold: f32,
    /// The size of the Gaussian kernel.
    /// Must be an odd number greater than 1.
    /// The larger the kernel, the more the image will be blurred.
    kernel_size: usize,
    /// The standard deviation of the Gaussian kernel.
    /// Must be greater than 0.0.
    /// The larger the sigma, the more the image will be blurred.
    sigma: f32,
}

impl CannyEdgeDetector {
    /// ## Constructor
    /// Create a new Canny edge detector with the given parameters.
    pub fn new(low_threshold: f32, high_threshold: f32, kernel_size: usize, sigma: f32) -> Self {
        CannyEdgeDetector {
            low_threshold,
            high_threshold,
            kernel_size,
            sigma,
        }
    }

    /// ## Gaussian blur
    /// Apply Gaussian blur to a grayscale image.
    ///
    /// The purpose of Gaussian blur is to reduce noise and detail in an image,
    /// often as a preprocessing step for other algorithms (e.g., edge detection).

    fn gaussian_blur(&self, image: &GrayImage) -> GrayImage {
        /// ### Generate a Gaussian kernel.
        ///
        /// The kernel is a square matrix that represents the Gaussian function,
        /// which is used to convolve with the input image to achieve the blurring effect.

        fn gaussian_kernel(size: usize, sigma: f32) -> Vec<Vec<f32>> {
            let mut kernel = vec![vec![0.0; size]; size];
            let s = 2.0 * sigma * sigma;
            let center = (size as isize) / 2;
            let mut sum = 0.0;

            for i in 0..size {
                for j in 0..size {
                    let x = i as isize - center;
                    let y = j as isize - center;
                    let value = ((-(x * x + y * y) as f32) / s).exp();
                    kernel[i][j] = value;
                    sum += value;
                }
            }

            for i in 0..size {
                for j in 0..size {
                    kernel[i][j] /= sum;
                }
            }

            kernel
        }

        /// ### Apply the kernel to the input image through convolution.
        ///
        /// Convolution is a mathematical operation that combines the kernel and the image,
        /// producing a new image where each pixel is the weighted sum of its neighbors.

        fn apply_kernel(image: &GrayImage, kernel: &[Vec<f32>]) -> GrayImage {
            let (width, height) = (image.width() as isize, image.height() as isize);
            let kernel_size = kernel.len();
            let offset = kernel_size / 2;

            GrayImage::from_fn(image.width(), image.height(), |x, y| {
                let mut sum = 0.0;

                for i in 0..kernel_size {
                    for j in 0..kernel_size {
                        let x_coord = x as isize - j as isize + offset as isize;
                        let y_coord = y as isize - i as isize + offset as isize;

                        if x_coord >= 0 && x_coord < width && y_coord >= 0 && y_coord < height {
                            let pixel_value =
                                image.get_pixel(x_coord as u32, y_coord as u32).0[0] as f32;
                            sum += pixel_value * kernel[i][j];
                        }
                    }
                }

                let pixel_value = sum.round().max(0.0).min(255.0) as u8;
                Luma([pixel_value])
            })
        }

        // Generate the Gaussian kernel based on the provided kernel size and sigma value.
        let kernel = gaussian_kernel(self.kernel_size, self.sigma);
        // Apply the kernel to the input image through convolution to produce the blurred image.
        apply_kernel(image, &kernel)
    }

    /// ## Sobel operator
    /// Apply the Sobel operator to a grayscale image to compute the gradient magnitude and direction.
    ///
    /// The Sobel operator is used for edge detection, highlighting areas in the image with rapid
    /// intensity changes. It can also be used as a preprocessing step for other image processing tasks.

    fn sobel_operator(&self, image: &GrayImage) -> (Array2<f32>, Array2<f32>) {
        let width = image.width() as usize;
        let height = image.height() as usize;
        let image_array = Array2::from_shape_fn((height, width), |(i, j)| {
            image.get_pixel(j as u32, i as u32)[0] as f32
        });

        // Sobel kernels for computing the gradients in the x and y directions.
        let gx = array![[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]];
        let gy = array![[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]];

        let mut gradient_magnitude = Array2::zeros((height, width));
        let mut gradient_direction = Array2::zeros((height, width));

        // Compute the gradient magnitude and direction for each pixel.
        for i in 1..height - 1 {
            for j in 1..width - 1 {
                let window = image_array.slice(s![i - 1..=i + 1, j - 1..=j + 1]);

                // Compute the gradients in the x and y directions using the Sobel kernels.
                let gradient_x: f32 = (gx.clone() * window).sum();
                let gradient_y: f32 = (gy.clone() * window).sum();

                // Calculate the magnitude and direction of the gradient.
                let magnitude = (gradient_x.powi(2) + gradient_y.powi(2)).sqrt();
                let direction = gradient_y.atan2(gradient_x);

                gradient_magnitude[[i, j]] = magnitude;
                gradient_direction[[i, j]] = direction;
            }
        }

        (gradient_magnitude, gradient_direction)
    }

    /// ## Non-maximum suppression
    /// Apply non-maximum suppression to the gradient magnitude and direction arrays.
    ///
    /// Non-maximum suppression is used to thin out the edges detected by the Sobel operator.
    /// It suppresses non-maximum gradient magnitudes by setting them to zero, keeping only the
    /// local maxima in the gradient direction.

    fn non_maximum_suppression(
        &self,
        gradient_magnitude: &ArrayView2<f32>,
        gradient_direction: &ArrayView2<f32>,
    ) -> Array2<f32> {
        let height = gradient_magnitude.dim().0;
        let width = gradient_magnitude.dim().1;

        let mut suppressed = Array2::zeros((height, width));

        // Iterate through the gradient magnitude and direction arrays,
        // applying non-maximum suppression to each pixel.
        for i in 1..height - 1 {
            for j in 1..width - 1 {
                let direction = gradient_direction[[i, j]];
                let magnitude = gradient_magnitude[[i, j]];

                // Quantize the gradient direction into one of four discrete angles (0, 1, 2, or 3).
                let angle = (((direction + std::f32::consts::PI) * 4.0)
                    / (2.0 * std::f32::consts::PI))
                    .round()
                    % 4.0;

                // Pick coordinates of the two neighboring pixels along the gradient direction.
                let ((y1, x1), (y2, x2)) = match angle as u8 {
                    0 => ((i - 1, j), (i + 1, j)),
                    1 => ((i - 1, j + 1), (i + 1, j - 1)),
                    2 => ((i, j - 1), (i, j + 1)),
                    3 => ((i - 1, j - 1), (i + 1, j + 1)),
                    _ => unreachable!(),
                };

                // Determine if the current pixel has a larger gradient magnitude
                // than its neighbors along the gradient direction.
                if magnitude > gradient_magnitude[[y1, x1]]
                    && magnitude > gradient_magnitude[[y2, x2]]
                {
                    suppressed[[i, j]] = magnitude;
                }
            }
        }

        // Return the thinned edge map after applying non-maximum suppression.
        suppressed
    }

    /// ## Double thresholding
    /// Apply double thresholding to the input image.
    ///
    /// Double thresholding is used to distinguish between strong and weak edges,
    /// which helps reduce false positives in edge detection.
    /// Pixels with gradient magnitudes:
    /// - above the high threshold are considered strong edges,
    /// - between the low and high thresholds are weak edges,
    /// - below the low threshold are not considered as edges at all.

    fn double_threshold(&self, image: &Array2<f32>) -> Array2<u8> {
        let height = image.dim().0;
        let width = image.dim().1;

        let mut thresholded = Array2::zeros((height, width));

        // Iterate through the input image and classify each pixel as strong, weak, or non-edge.
        for ((i, j), pixel) in image.indexed_iter() {
            thresholded[[i, j]] = match *pixel {
                // Classify as strong edge
                p if p >= self.high_threshold => 2,
                // Classify as weak edge
                p if p >= self.low_threshold => 1,
                // Classify as non-edge
                _ => 0,
            };
        }

        // Return the image with pixels classified as strong, weak, or non-edge.
        thresholded
    }

    /// ## Edge tracking by hysteresis
    /// Perform edge tracking by hysteresis on the input image.
    ///
    /// Edge tracking by hysteresis is used to convert weak edges into strong edges
    /// if they are connected to strong edges, thus further reducing false positives
    /// in edge detection.

    fn edge_tracking_by_hysteresis(&self, image: &Array2<u8>) -> Array2<u8> {
        let height = image.dim().0;
        let width = image.dim().1;

        let mut final_edges = Array2::zeros((height, width));

        /// ### Recursive depth-first search (DFS) function
        /// Starting from a strong edge, recursively visit all weak edges connected to it,
        /// converting them to strong edges.
        /// Emulates a depth-first search (DFS) starting from the current pixel.
        ///
        /// This is necessary to reduce false positives in edge detection.
        /// If a weak edge is not connected to a strong edge, it is not considered an edge at all.

        fn dfs_visit(y: usize, x: usize, image: &Array2<u8>, final_edges: &mut Array2<u8>) {
            let neighbors = [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ];

            // Iterate through the 8 neighboring pixels.
            for (dy, dx) in &neighbors {
                let ny = y as isize + dy;
                let nx = x as isize + dx;

                // Check if the neighboring pixel is within the image bounds.
                if ny >= 0 && nx >= 0 && ny < image.dim().0 as isize && nx < image.dim().1 as isize
                {
                    let (ny, nx) = (ny as usize, nx as usize);
                    // If the neighboring pixel is a weak edge and not visited, convert it to a strong edge.
                    if image[[ny, nx]] == 1 && final_edges[[ny, nx]] == 0 {
                        final_edges[[ny, nx]] = 255;
                        dfs_visit(ny, nx, image, final_edges);
                    }
                }
            }
        }

        // Iterate through the input image and perform edge tracking by hysteresis.
        for ((y, x), pixel) in image.indexed_iter() {
            // If the current pixel is a strong edge, convert it to a strong edge.
            if *pixel == 2 && final_edges[[y, x]] == 0 {
                final_edges[[y, x]] = 255;
                dfs_visit(y, x, image, &mut final_edges);
            }
        }

        // Return the image with the final strong edges.
        final_edges
    }

    /// Invert the pixel values of a grayscale image.
    ///
    /// Inverting a grayscale image can be useful for processing, visualization, or
    /// enhancing specific features. Inverted images may also be more suitable for certain
    /// algorithms that rely on high pixel values to represent important information.

    fn invert_gray_image(image: &GrayImage) -> GrayAlphaImage {
        let width = image.width();
        let height = image.height();

        GrayAlphaImage::from_fn(width, height, |x, y| {
            let pixel = image.get_pixel(x, y).0[0];
            let inverted_pixel = 255 - pixel;
            LumaA([inverted_pixel, 255]) // if inverted_pixel == 255 { 0 } else { 255 }])
        })
    }

    /// ## Main function to perform edge detection
    /// Detect edges in the input image using the Canny edge detection algorithm.
    ///
    /// The input image is converted to grayscale, blurred, and then the gradient
    /// of the image is computed using the Sobel operator. Non-maximum suppression
    /// is then used to thin the edges, followed by double thresholding to
    /// distinguish between strong and weak edges. Finally, edge tracking by
    /// hysteresis is used to connect edge segments.
    ///
    /// The final output is a grayscale image with strong edges represented by
    /// black pixels and weak edges represented by gray pixels.

    pub fn process_image(&self, image: &DynamicImage) -> DynamicImage {
        // Step 1: Convert the input image to grayscale
        let image = image.grayscale().to_luma8();

        // Step 2: Apply a Gaussian blur to reduce noise and smooth the image
        let blurred_image = self.gaussian_blur(&image);

        // Step 3: Compute the gradient of the image using the Sobel operator
        let (gradient_magnitude, gradient_direction) = self.sobel_operator(&blurred_image);

        // Step 4: Perform non-maximum suppression to thin the edges
        let suppressed_image =
            self.non_maximum_suppression(&gradient_magnitude.view(), &gradient_direction.view());

        // Step 5: Apply double thresholding to determine potential edges
        let thresholded_image = self.double_threshold(&suppressed_image);

        // Step 6: Perform edge tracking by hysteresis to connect edge segments
        let final_edges = self.edge_tracking_by_hysteresis(&thresholded_image);

        // Convert the final edge map to a GrayImage
        let image = GrayImage::from_raw(image.width(), image.height(), final_edges.into_raw_vec())
            .expect("Failed to convert final edges to GrayImage");

        // Invert the image to make the edges black and the background white
        let image = CannyEdgeDetector::invert_gray_image(&image);
        DynamicImage::ImageLumaA8(image)
    }
}

use std::env;
use std::process;

pub fn main() {
    // Read command-line arguments
    let args: Vec<String> = env::args().collect();

    // Check if the image file argument is provided
    if args.len() != 2 {
        eprintln!("Usage: {} <image_file>", args[0]);
        process::exit(1);
    }

    let image_file = &args[1];

    let image = image::open(image_file).unwrap();
    let canny = CannyEdgeDetector::new(5.0, 10.0, 1, 1.0);
    let edges = canny.process_image(&image);
    edges.save(image_file).unwrap();
}
