#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace cv::ml;
using namespace std;

int numImage = 140;
int imageData = 22;

int main()
{
    float trainingDatas[140][22];
    float labels[140];

    cv::Ptr<cv::ml::TrainData> raw_data = cv::ml::TrainData::loadFromCSV(
        "C:/Users/TienYu/Desktop/OpenCV_demo/opencv_judge_image_saveData/build/Debug/trainingDataMat.csv", 0, -2, 0);
    cv::Mat mlData = raw_data->getSamples();

    cv::Ptr<cv::ml::TrainData> raw_label = cv::ml::TrainData::loadFromCSV(
        "C:/Users/TienYu/Desktop/OpenCV_demo/opencv_judge_image_saveData/build/Debug/labelsMat.csv", 0, -2, 0);
    cv::Mat mlLabel = raw_label->getSamples();

    //cout << "mat 2 arr"<<endl;

    for (int i = 0; i < mlData.rows; i++)
    {
        for (int j = 0; j < mlData.cols; j++)
        {
            trainingDatas[i][j] = mlData.at<float_t>(i, j);
            //cout << trainingDatas[i][j] << " ";
        }
        //cout << endl;
    }
    //cout << "label = " << endl;
    for (int i = 0; i < mlLabel.rows; i++)
    {
        labels[i] = mlLabel.at<float_t>(i);
        //cout << labels[i] << " " << endl;
    }

    Mat trainingDataMat(numImage, imageData, CV_32FC1, trainingDatas);
    Mat labelsMat(numImage, 1, CV_32FC1, labels);

    Ptr<ANN_MLP> ann = ANN_MLP::create();
    Mat layerSizes = (Mat_<int>(1, 3) << imageData, 140, 1);
    ann->setLayerSizes(layerSizes);
    ann->setTrainMethod(ANN_MLP::BACKPROP);
    ann->setActivationFunction(ANN_MLP::SIGMOID_SYM);

    Ptr<TrainData> tData = TrainData::create(trainingDataMat, ROW_SAMPLE, labelsMat);

    cout << "Start training" << endl;
    ann->train(tData);
    cout << "Training completed" << endl;
    ann->save("C:/Users/TienYu/Desktop/opencv_final/ANN_model.xml");
    int error = 0;
    cout << "Start predict" << endl;
    
    for (int i = 0; i < numImage; i++)
    {
        Mat sampleMat = (Mat_<float>(1, imageData) << trainingDatas[i][0], trainingDatas[i][1], trainingDatas[i][2], trainingDatas[i][3], trainingDatas[i][4], trainingDatas[i][5], trainingDatas[i][6], trainingDatas[i][7], trainingDatas[i][8], trainingDatas[i][9], trainingDatas[i][10], trainingDatas[i][11], trainingDatas[i][12], trainingDatas[i][13], trainingDatas[i][14], trainingDatas[i][15], trainingDatas[i][16], trainingDatas[i][17], trainingDatas[i][18], trainingDatas[i][19], trainingDatas[i][20], trainingDatas[i][21]);

        Mat response_mat;
        // float response = model->predict(sampleMat);

        ann->predict(sampleMat, response_mat);

        float response = response_mat.ptr<float>(0)[0];

        if (response < 0.5)
            response = 0;

        else if (response >= 0.5)
            response = 1;

        if (response != labels[i])
        {
            cout << "NO. " << i + 1 << " Incorrect" << endl;
            cout << "Response = " << response << " , labels = " << labels[i] << endl;
            cout << "--------------------" << endl;
            error++;
        }
    }
    float ClassificationAccuracy = 1 - ((float)error / (float)numImage);

    cout << "ClassificationAccuracy : " << ClassificationAccuracy * 100 << "%" << endl;
    
}
