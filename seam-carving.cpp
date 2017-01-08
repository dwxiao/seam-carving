#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <vector>


using namespace cv;
using namespace std;

enum SeamDirection { VERTICAL, HORIZONTAL };

void printMatrix(const Mat &M) {
    cout << "M =" << endl << M << endl << endl;
}

Mat createEnergyImage(Mat &image) {
    //Mat image;
    Mat image_blur;
    Mat image_gray;
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Mat grad, energy_image;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    
    // read in a file and check to see if it is valid or not
    /*
    image = imread(filename);
    if (image.empty()) {
        cout << "Unable to load image" << endl;
        exit(EXIT_FAILURE);
    }
    */
    // apply a gaussian blur to reduce noise
    GaussianBlur(image, image_blur, Size(3,3), 0, 0, BORDER_DEFAULT);
    
    // convert to grayscale
    cvtColor(image_blur, image_gray, CV_BGR2GRAY);
    
    // use Sobel to calculate the gradient of the image in the x and y direction
    Sobel(image_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    Sobel(image_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
    
    // use Scharr to calculate the gradient of the image in the x and y direction
    //Scharr(image_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT);
    //Scharr(image_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT);

    // convert gradients to abosulte versions of themselves
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    
    /// total gradient (approx)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
    
    // convert the default values to double precision
    grad.convertTo(energy_image, CV_64F, 1.0/255.0);
    
    // create and show the newly created energy image
    //namedWindow("Energy Image", CV_WINDOW_AUTOSIZE); imshow("Energy Image", energy_image); waitKey(0);
    
    return energy_image;
}

Mat createCumulativeEnergyMap(Mat &energy_image, int seam_direction) {
    double a,b,c;
    
    // get the numbers of rows and columns in the image
    int rowsize = energy_image.rows;
    int colsize = energy_image.cols;
    
    // initialize the map with zeros
    Mat cumulative_energy_map = Mat(rowsize, colsize, CV_64F, double(0));
    
    // copy the first row
    if (seam_direction == VERTICAL) energy_image.row(0).copyTo(cumulative_energy_map.row(0));
    else if (seam_direction == HORIZONTAL) energy_image.col(0).copyTo(cumulative_energy_map.col(0));
    
    // dynamic programming ftw
    if (seam_direction == VERTICAL) {
        for (int row = 1; row < rowsize; row++) {
            for (int col = 0; col < colsize; col++) {
                a = cumulative_energy_map.at<double>(row - 1, max(col - 1, 0));
                b = cumulative_energy_map.at<double>(row - 1, col);
                c = cumulative_energy_map.at<double>(row - 1, min(col + 1, colsize - 1));
                
                cumulative_energy_map.at<double>(row, col) = energy_image.at<double>(row, col) + std::min(a, min(b, c));
            }
        }
    }
    else if (seam_direction == HORIZONTAL) {
        for (int col = 1; col < colsize; col++) {
            for (int row = 0; row < rowsize; row++) {
                a = cumulative_energy_map.at<double>(max(row - 1, 0), col - 1);
                b = cumulative_energy_map.at<double>(row, col - 1);
                c = cumulative_energy_map.at<double>(min(row + 1, rowsize - 1), col - 1);
                
                cumulative_energy_map.at<double>(row, col) = energy_image.at<double>(row, col) + std::min(a, min(b, c));
            }
        }
    }

    // convert to color scale (similar to MATLAB's imagesc()) -- only used during dev/demo purposes
    Mat test;
    double Cmin;
    double Cmax;
    cv::minMaxLoc(cumulative_energy_map, &Cmin, &Cmax);
    float scale = 255.0 / (Cmax - Cmin);
    cumulative_energy_map.convertTo(test, CV_8UC1, scale);
    cv::Mat result_test;
    applyColorMap(test, result_test, cv::COLORMAP_JET);
    
    // create an show the newly created cumulative energy map
    //namedWindow("Cumulative Energy Map", CV_WINDOW_AUTOSIZE); imshow("Cumulative Energy Map", result_test); waitKey(0);
    
    return cumulative_energy_map;
}

vector<int> findOptimalSeam(Mat &cumulative_energy_map, int seam_direction) {
    // declare variables
    double a,b,c;
    int offset = 0;
    vector<int> path;
    double min_val, max_val;
    Point min_pt, max_pt;
    
    // get the number of rows and columns in the cumulative energy map
    int rowsize = cumulative_energy_map.rows;
    int colsize = cumulative_energy_map.cols;
    
    if (seam_direction == VERTICAL) {
        // copy the data from the last row of the cumulative energy map
        Mat row = cumulative_energy_map.row(rowsize - 1);
    
        // get min and max values and locations
        minMaxLoc(row, &min_val, &max_val, &min_pt, &max_pt);
        
        // initialize the path vector
        path.resize(rowsize);
        int min_index = min_pt.x;
        path[rowsize - 1] = min_index;
        
        // starting from the bottom, look at the three adjacent pixels above current pixel, choose the minimum of those and add to the path
        for (int i = rowsize - 2; i >= 0; i--) {
            a = cumulative_energy_map.at<double>(i, max(min_index - 1, 0));
            b = cumulative_energy_map.at<double>(i, min_index);
            c = cumulative_energy_map.at<double>(i, min(min_index + 1, colsize - 1));
            
            if (min(a,b) > c) {
                offset = 1;
            }
            else if (min(a,c) > b) {
                offset = 0;
            }
            else if (min(b, c) > a) {
                offset = -1;
            }
            
            min_index += offset;
            min_index = min(max(min_index, 0), colsize - 1); // take care of edge cases
            path[i] = min_index;
        }
    }
    
    else if (seam_direction == HORIZONTAL) {
        // copy the data from the last column of the cumulative energy map
        Mat col = cumulative_energy_map.col(colsize - 1);
        
        // get min and max values and locations
        minMaxLoc(col, &min_val, &max_val, &min_pt, &max_pt);
        
        // initialize the path vector
        path.resize(colsize);
        int min_index = min_pt.y;
        path[colsize - 1] = min_index;
        
        // starting from the right, look at the three adjacent pixels to the left of current pixel, choose the minimum of those and add to the path
        for (int i = colsize - 2; i >= 0; i--) {
            a = cumulative_energy_map.at<double>(max(min_index - 1, 0), i);
            b = cumulative_energy_map.at<double>(min_index, i);
            c = cumulative_energy_map.at<double>(min(min_index + 1, rowsize - 1), i);
            
            if (min(a,b) > c) {
                offset = 1;
            }
            else if (min(a,c) > b) {
                offset = 0;
            }
            else if (min(b, c) > a) {
                offset = -1;
            }
            
            min_index += offset;
            min_index = min(max(min_index, 0), rowsize - 1); // take care of edge cases
            path[i] = min_index;
        }
    }
    
    return path;
}

void showPath(Mat &image, vector<int> path, int seam_direction) {
    // read in a file and check to see if it is valid or not
    //Mat image = createEnergyImage(filename);
    
    // loop through the image and change all pixels in the path to white
    if (seam_direction == VERTICAL) {
        for (int i = 0; i < image.rows; i++) {
            image.at<double>(i,path[i]) = 1;
        }
    }
    else if (seam_direction == HORIZONTAL) {
        for (int i = 0; i < image.cols; i++) {
            image.at<double>(path[i],i) = 1;
        }
    }
    // display the seam on top of the energy image
    namedWindow("Seam on Energy Image", CV_WINDOW_AUTOSIZE); imshow("Seam on Energy Image", image); waitKey(0);
}

void reduce(Mat &image, vector<int> path, int seam_direction) {
    // read in a file and check to see if it is valid or not
    /*
    Mat image = imread(filename);
    if (image.empty()) {
        cout << "Unable to load image" << endl;
        exit(EXIT_FAILURE);
    }
    */
    // get the number of rows and columns in the image
    int rowsize = image.rows;
    int colsize = image.cols;
    
    // create a 1x1x3 dummy matrix to add onto the tail of a new row to maintain image dimensions and mark for deletion
    Mat dummy(1, 1, CV_8UC3, Vec3b(0, 0, 0));
    
    if (seam_direction == VERTICAL) { // reduce the width
        for (int i = 0; i < rowsize; i++) {
            // take all pixels to the left and right of marked pixel and store them in appropriate subrow variables
            Mat new_row;
            Mat lower = image.rowRange(i, i + 1).colRange(0, path[i]);
            Mat upper = image.rowRange(i, i + 1).colRange(path[i] + 1, colsize);
            
            // merge the two subrows and dummy matrix/pixel into a full row
            if (!lower.empty() && !upper.empty()) {
                hconcat(lower, upper, new_row);
                hconcat(new_row, dummy, new_row);
            }
            else {
                if (lower.empty()) {
                    hconcat(upper, dummy, new_row);
                }
                else if (upper.empty()) {
                    hconcat(lower, dummy, new_row);
                }
            }
            // take the newly formed row and place it into the original image
            new_row.copyTo(image.row(i));
        }
        // clip the right-most side of the image
        image = image.colRange(0, colsize - 1);
    }
    else if (seam_direction == HORIZONTAL) { // reduce the height
        for (int i = 0; i < colsize; i++) {
            // take all pixels to the top and bottom of marked pixel and store the in appropriate subcolumn variables
            Mat new_col;
            Mat lower = image.colRange(i, i + 1).rowRange(0, path[i]);
            Mat upper = image.colRange(i, i + 1).rowRange(path[i] + 1, rowsize);
            
            // merge the two subcolumns and dummy matrix/pixel into a full row
            if (!lower.empty() && !upper.empty()) {
                vconcat(lower, upper, new_col);
                vconcat(new_col, dummy, new_col);
            }
            else {
                if (lower.empty()) {
                    vconcat(upper, dummy, new_col);
                }
                else if (upper.empty()) {
                    vconcat(lower, dummy, new_col);
                }
            }
            // take the newly formed column and place it into the original image
            new_col.copyTo(image.col(i));
        }
        // clip the bottom-most side of the image
        image = image.rowRange(0, rowsize - 1);
    }
    
    //namedWindow("Reduced Image", CV_WINDOW_AUTOSIZE); imshow("Reduced Image", image); waitKey(0);
}

int main() {
    string filename = "prague.jpg";
    int seam_direction = VERTICAL;
    
    Mat image = imread(filename);
    if (image.empty()) {
        cout << "Unable to load image" << endl;
        exit(EXIT_FAILURE);
    }
    
    namedWindow("Original Image", CV_WINDOW_AUTOSIZE); imshow("Original Image", image);
    
    for (int i = 0; i < 101; i++) {
        Mat energy_image = createEnergyImage(image);
        Mat cumulative_energy_map = createCumulativeEnergyMap(energy_image, seam_direction);
        vector<int> path = findOptimalSeam(cumulative_energy_map, seam_direction);
        reduce(image, path, seam_direction);
    }
    
    cout << image.rows << "x" << image.cols << endl;
    
    namedWindow("Reduced Image", CV_WINDOW_AUTOSIZE); imshow("Reduced Image", image); waitKey(0);
    
    return 0;
}