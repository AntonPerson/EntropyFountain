# Canny Edge Detector in Rust

This Rust project contains an implementation of the Canny Edge Detection algorithm, a popular edge detection algorithm used in image processing. The algorithm is composed of several steps including Gaussian blur, gradient calculation, non-maximum suppression, and hysteresis thresholding.

## Table of Contents
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
- [Usage](#usage)
    - [Example Code](#example-code)
- [Contributing](#contributing)
- [References](#references)
- [License](#license)

## Getting Started

### Prerequisites

- Rust: You need to have Rust installed on your system. If you don't have Rust installed, you can use rustup (Rust toolchain manager). Visit [the official Rust website](https://www.rust-lang.org/tools/install) to install rustup.

### Installation

1. Clone the repository (replace `<USER>` and `<PROJECT>`):
   ```
   git clone https://github.com/<USER>/<PROJECT>.git
   cd <PROJECT>/canny-edge-detector
   ```

2. Build the project:
   ```
   cargo build --release
   ```

This will create an executable in the `./target/release` directory.

## Usage

After building the project, you can use the command line application as follows:

```
./target/release/canny_edge_detector <path_to_image_file>
```

This command reads an image, processes it using the Canny Edge Detector algorithm, and saves the resulting image with edges highlighted. Your input image will be overwritten.

An alternative to quickly run the code with debug settings (= less optimized) is as usual:

   ```
   cargo run -- <path_to_image_file>
   ```


### Example Code

Here's how you can use the `CannyEdgeDetector` struct in your Rust program:

```rust
use canny_edge_detector::CannyEdgeDetector;

let input = image::open("input.png").unwrap();
let canny = CannyEdgeDetector::new(50.0, 100.0, 5, 1.0);
let edges = canny.process_image(&input);
edges.save("output.png").unwrap();
```

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**. Here's how you can contribute:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## References

- Canny, J. (1986). A computational approach to edge detection. IEEE Transactions on pattern analysis and machine intelligence, 8(6), 679-698.
- [Wikipedia - Canny edge detector](https://en.wikipedia.org/wiki/Canny_edge_detector)
- [Canny Edge Detector - Youtube](https://www.youtube.com/watch?v=sRFM5IEqR2w)
- [Sobel Operator - Youtube](https://www.youtube.com/watch?v=uihBwtPIBxM)

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgements

- [The Rust Programming Language](https://doc.rust-lang.org/book/) - The book on Rust; a great resource to get started!
- [Rust by Example](https://doc.rust-lang.org/stable/rust-by-example/) - A collection of runnable examples that illustrate various Rust concepts and standard libraries.
- [image](https://crates.io/crates/image) - A Rust library providing basic image processing functions.
- [ndarray](https://crates.io/crates/ndarray) - A Rust library for N-dimensional arrays