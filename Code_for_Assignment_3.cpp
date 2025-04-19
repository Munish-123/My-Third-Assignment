#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

// Custom convolution function
Mat customConvolve(const Mat& image, const Mat& kernel) {
    int h = image.rows;
    int w = image.cols;
    int kh = kernel.rows;
    int kw = kernel.cols;
    int pad = kh / 2;

    // Pad the image
    Mat padded;
    copyMakeBorder(image, padded, pad, pad, pad, pad, BORDER_CONSTANT, Scalar(0));

    Mat output = Mat::zeros(image.size(), CV_32F);

    // Perform convolution
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            Rect roi(j, i, kw, kh);
            Mat region = padded(roi);
            output.at<float>(i, j) = sum(region.mul(kernel))[0];
        }
    }

    // Clip and convert to 8-bit
    Mat result;
    output.convertTo(result, CV_8U, 1, 0);
    return result;
}

int main() {
    string inputPath, outputFolder;

    // Ask user for input image path
    cout << "Enter the path of the grayscale image (e.g., C:/Users/YourName/sample.png): ";
    getline(cin, inputPath);

    // Load image in grayscale
    Mat image = imread(inputPath, IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "❌ Could not find or open the image at the given path!" << endl;
        return -1;
    }

    // Ask user for output folder
    cout << "Enter the folder path where you want to save the results (e.g., C:/Users/YourName/Images): ";
    getline(cin, outputFolder);

    // Define kernels
    Mat blurKernel = (Mat_<float>(3, 3) <<
        1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
        1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
        1 / 9.0f, 1 / 9.0f, 1 / 9.0f);

    Mat sharpenKernel = (Mat_<float>(3, 3) <<
         0, -1,  0,
        -1,  5, -1,
         0, -1,  0);

    Mat edgeKernel = (Mat_<float>(3, 3) <<
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1);

    // Apply custom convolution
    Mat blurred   = customConvolve(image, blurKernel);
    Mat sharpened = customConvolve(image, sharpenKernel);
    Mat edges     = customConvolve(image, edgeKernel);

    // Compose full output paths
    string blurPath    = outputFolder + "/blur_output.png";
    string sharpenPath = outputFolder + "/sharpen_output.png";
    string edgePath    = outputFolder + "/edge_output.png";

    // Save the results
    imwrite(blurPath, blurred);
    imwrite(sharpenPath, sharpened);
    imwrite(edgePath, edges);

    cout << "✅ Done! Check your folder (" << outputFolder << ") for:" << endl;
    cout << " - blur_output.png" << endl;
    cout << " - sharpen_output.png" << endl;
    cout << " - edge_output.png" << endl;

    return 0;
}
