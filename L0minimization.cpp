#include <opencv2/opencv.hpp>
#include <Eigen/Sparse>

// Optimization params
float lambda = 0.01;
float beta0 = 2*lambda;
float beta_max = 100000;
float kappa = 2.0;
bool exact = false;

void buildGradientMatrix(Eigen::SparseMatrix<float> &G, 
                         const int rows,
                         const int cols,
                         const std::vector<std::pair<int, float> > x_indices, 
                         const std::vector<std::pair<int, float> > y_indices
                         )
{
    int num_of_variables = rows*cols;
    std::vector<Eigen::Triplet<float> > coeffcients;
    bool compute_x = x_indices.empty() ? false : true;
    bool compute_y = y_indices.empty() ? false : true;

    G = Eigen::SparseMatrix<float>(num_of_variables, num_of_variables);

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            int index = i*cols+j;
            int n_index, num_indices;
            if(compute_x){
                num_indices = x_indices.size();
                for(int k=0; k<num_indices; k++){
                    n_index = index + x_indices[k].first;                    
                    if(n_index >= num_of_variables){
                        continue;
                    }
                    coeffcients.push_back(Eigen::Triplet<float>(index, n_index, x_indices[k].second));
                }
            }
            if(compute_y){
                num_indices = y_indices.size();
                for(int k=0; k<num_indices; k++){                    
                    n_index = (i+y_indices[k].first)*cols+j;
                    if(n_index >= num_of_variables){
                        continue;
                    }
                    coeffcients.push_back(Eigen::Triplet<float>(index, n_index, y_indices[k].second));
                }                    
            }
        }
    }
    G.setFromTriplets(coeffcients.begin(), coeffcients.end());
}
void constructSparseIdentityMatrix(Eigen::SparseMatrix<float> &mat, const int &num_of_variables){
    mat = Eigen::SparseMatrix<float>(num_of_variables, num_of_variables);
    std::vector<Eigen::Triplet<float> > coeffcients;
    for(int i=0; i<num_of_variables; i++){
        coeffcients.push_back(Eigen::Triplet<float>(i, i, 1.0f));        
    }
    mat.setFromTriplets(coeffcients.begin(), coeffcients.end());
}
void vec2CvMat(const Eigen::VectorXf &vec, cv::Mat &mat, const int &rows, const int &cols){
    mat = cv::Mat(cols, rows, CV_32FC1);
    for(int i=0; i<rows; i++){
        float *ptr = reinterpret_cast<float*>(mat.data+mat.step*i);
        for(int j=0; j<cols; j++){
            *ptr = vec[i*cols+j];
            ++ptr;
        }
    }   
}
void cvMat2Vec(const cv::Mat &mat, Eigen::VectorXf &vec){
    int rows = mat.rows;
    int cols = mat.cols;
    vec = Eigen::VectorXf::Zero(rows*cols);

    for(int i=0; i<rows; i++){
        float *ptr = reinterpret_cast<float*>(mat.data+mat.step*i);
        for(int j=0; j<cols; j++){
            vec[i*cols+j] = *ptr;
            ++ptr;
        }
    }    
}

void computeGradient(const cv::Mat &mat, cv::Mat &grad_x, cv::Mat &grad_y){
    int rows = mat.rows;
    int cols = mat.cols;
    grad_x = cv::Mat::zeros(rows, cols, CV_32FC1);
    grad_y = cv::Mat::zeros(rows, cols, CV_32FC1);    

    for(int i=0; i<rows-1; i++){        
        for(int j=0; j<cols-1; j++){        
            grad_x.at<float>(i, j) = mat.at<float>(i, j) - mat.at<float>(i, j+1);
            grad_y.at<float>(i, j) = mat.at<float>(i, j) - mat.at<float>(i+1, j);
        }
    }
}

void computeS(cv::Mat &S, 
              const cv::Mat &I,
              const cv::Mat &H,
              const cv::Mat &V,
              const float &beta
              )
{
    int rows = S.rows;
    int cols = S.cols;
    int num_of_variables = rows*cols;

    Eigen::VectorXf S_vec, I_vec, H_vec, V_vec;
    cvMat2Vec(I, I_vec);
    cvMat2Vec(H, H_vec);
    cvMat2Vec(V, V_vec);

    Eigen::SparseMatrix<float> GX, GY;
    std::vector<std::pair<int, float> > indices;
    indices.push_back(std::pair<int, float>(0, 1.0f));
    indices.push_back(std::pair<int, float>(1, -1.0f));
    buildGradientMatrix(GX, rows, cols, indices, std::vector<std::pair<int, float> >());
    buildGradientMatrix(GY, rows, cols, std::vector<std::pair<int, float> >(), indices);

    // build linear system Ax=b
    Eigen::SparseMatrix<float> E;
    constructSparseIdentityMatrix(E, num_of_variables);
    Eigen::SparseMatrix<float> A = beta*(GX.transpose()*GX+GY.transpose()*GY)+E;

    Eigen::VectorXf b = I_vec + beta*(GX.transpose()*H_vec+GY.transpose()*V_vec);

    // solve linear system
    if(exact){
        Eigen::SimplicialLLT<Eigen::SparseMatrix<float> > solver;
        solver.compute(A);
        if(solver.info()!=Eigen::Success) {
            std::cout << "decomposition failed" << std::endl;
        }    
        S_vec = solver.solve(b);
    }
    else{
        Eigen::ConjugateGradient<Eigen::SparseMatrix<float> > solver;
        S_vec = solver.compute(A).solve(b);        
    }

    vec2CvMat(S_vec, S, rows, cols);
}

void optimize(cv::Mat &S, cv::Mat &I, cv::Mat &H, cv::Mat &V, float &beta){

    int rows = S.rows;
    int cols = S.cols;

    /// Generate grad_x and grad_y
    cv::Mat grad_x, grad_y;

    // Compute Gradient
    computeGradient(S, grad_x, grad_y);

    // Computing h, v
    for(int j=0; j<rows; j++){
        for(int i=0; i<cols; i++){

            float gx = grad_x.at<float>(j, i);
            float gy = grad_y.at<float>(j, i);
            float val = gx*gx + gy*gy;        

            if(val < lambda/beta){
                H.at<float>(j, i) = V.at<float>(j, i) = 0;
            }
            else{          
                H.at<float>(j, i) = gx;
                V.at<float>(j, i) = gy;
            }      
        }            
    }
    // Computing s 
    computeS(S, I, H, V, beta);
}

cv::Mat minimizeL0Gradient(const cv::Mat &src){
    std::vector<cv::Mat> src_channels;
    cv::split(src, src_channels);    

    int num_of_channels = src_channels.size();
    std::vector<cv::Mat> S_channels(num_of_channels), I_channels(num_of_channels), S_U8_channels(num_of_channels);
    for(int i=0; i<num_of_channels; i++){
        src_channels[i].convertTo(I_channels[i], CV_32FC1);
        I_channels[i] *= 1./255;
        I_channels[0].copyTo(S_channels[i]);            
    }

    // initialize
    cv::Mat H, V, S;
    float beta = beta0;
    int count = 0;    
    H = cv::Mat(src.rows, src.cols, CV_32FC1);
    V = cv::Mat(src.rows, src.cols, CV_32FC1);

    // main loop
    while(beta < beta_max){

        // minimize L0 gradient
        for(int i=0; i<num_of_channels; i++){
            optimize(S_channels[i], I_channels[i], H, V, beta);
        }
        // Update param
        beta = beta*kappa;
        std::cout << "iteration #" << count++ << " beta: " << beta << std::endl;

        for(int i=0; i<num_of_channels; i++){
            cv::convertScaleAbs(S_channels[i], S_U8_channels[i], 255.0);
        }        
        cv::merge(S_U8_channels, S);
        cv::imshow("S", S);
        cv::waitKey(0);        
    }
    return S;
}

std::string usage = "./L0minimization src_img exact(0 or 1, default is 0)";

int main(int argc, char** argv){

    if(argc != 2 && argc != 3){
        std::cout <<"usage: " << usage << std::endl;
        return -1;
    }
    if(argc == 3){
        exact = atoi(argv[2]) == 1? true : false;
    }

    cv::Mat img = cv::imread(argv[1], 1);
    if(img.empty()){
        std::cout << "can't read input image " << std::endl;
        return -1;
    }

    cv::imshow("input", img);
    cv::Mat result = minimizeL0Gradient(img);
    cv::imshow("result", result);
    cv::waitKey(0);
    std::stringstream ss;
    ss << "result";
    if(exact){
        ss << "_exact.png";
    }
    else{
        ss << "_approx.png";
    }
    cv::imwrite(ss.str(), result);

    return 0;
}
