#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include "samples_utility.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;
using namespace dnn;

/* Automatic object detection code was modified based on code provided by 
 * https://github.com/spmallick/learnopencv/blob/master/ObjectDetection-YOLO/object_detection_yolo.cpp
 * Object detection is conducted using YOLO3
 */

// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;  // Width of network's input image
int inpHeight = 416; // Height of network's input image
vector<string> classes;

// Remove the bounding boxes with low confidence using non-maxima suppression
Rect postprocess(Mat& frame, const vector<Mat>& out);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net);


static const char* keys =
{ "{@tracker_algorithm | | Tracker algorithm }"
    "{@video_name      | | video name        }"
    "{@start_frame     |0| Start frame       }"
    "{@bounding_frame  |0,0,0,0| Initial bounding frame}"};

static void help()
{
  cout << "\nThis example shows the functionality of \"Long-term optical tracking API\""
       "-- pause video [p] and draw a bounding box around the target to start the tracker\n"
       "Example of <video_name> is in opencv_extra/testdata/cv/tracking/\n"
       "Call:\n"
       "./tracker <tracker_algorithm> <video_name> <start_frame> [<bounding_frame>]\n"
       "tracker_algorithm can be: MIL, BOOSTING, MEDIANFLOW, TLD, KCF, GOTURN, MOSSE.\n"
       << endl;

  cout << "\n\nHot keys: \n"
       "\tq - quit the program\n"
       "\tp - pause video\n";
}

int main( int argc, char** argv ){
  CommandLineParser parser( argc, argv, keys );

  // Load names of classes
  string classesFile = "coco.names";
  ifstream ifs(classesFile.c_str());
  string line;
  while (getline(ifs, line)) classes.push_back(line);
  
  // Give the configuration and weight files for the model
  String modelConfiguration = "yolov3.cfg";
  String modelWeights = "yolov3.weights";

  // Load the network
  Net net = readNetFromDarknet(modelConfiguration, modelWeights);
  net.setPreferableBackend(DNN_BACKEND_OPENCV);
  net.setPreferableTarget(DNN_TARGET_CPU);

  String tracker_algorithm = parser.get<String>( 0 );
  String video_name = parser.get<String>( 1 );
  int start_frame = parser.get<int>( 2 );

  if( tracker_algorithm.empty() || video_name.empty() )
  {
    help();
    return -1;
  }

  int coords[4]={0,0,0,0};
  bool initBoxWasGivenInCommandLine=false;
  {
      String initBoundingBox=parser.get<String>(3);
      for(size_t npos=0,pos=0,ctr=0;ctr<4;ctr++){
        npos=initBoundingBox.find_first_of(',',pos);
        if(npos==string::npos && ctr<3){
           printf("bounding box should be given in format \"x1,y1,x2,y2\",where x's and y's are integer cordinates of opposed corners of bdd box\n");
           printf("got: %s\n",initBoundingBox.substr(pos,string::npos).c_str());
           printf("manual selection of bounding box will be employed\n");
           break;
        }
        int num=atoi(initBoundingBox.substr(pos,(ctr==3)?(string::npos):(npos-pos)).c_str());
        if(num<=0){
           printf("bounding box should be given in format \"x1,y1,x2,y2\",where x's and y's are integer cordinates of opposed corners of bdd box\n");
           printf("got: %s\n",initBoundingBox.substr(pos,npos-pos).c_str());
           printf("manual selection of bounding box will be employed\n");
           break;
        }
        coords[ctr]=num;
        pos=npos+1;
      }
      if(coords[0]>0 && coords[1]>0 && coords[2]>0 && coords[3]>0){
          initBoxWasGivenInCommandLine=true;
      }
  }

  //open the capture
  string str, outputFile;
  outputFile = video_name;
  outputFile.replace(outputFile.end()-4, outputFile.end(), "_yolo_out_cpp.avi");
  VideoCapture cap;
  VideoWriter video;
  cap.open( video_name );
  cap.set( CAP_PROP_POS_FRAMES, start_frame );

  if( !cap.isOpened() )
  {
    help();
    cout << "***Could not initialize capturing...***\n";
    cout << "Current parameter's value: \n";
    parser.printMessage();
    return -1;
  }
  
  double freq = getTickFrequency();
  Mat frame, blob;
  namedWindow( "Tracking API", 1 );

  Mat image;
  Rect2d boundingBox;
  bool paused = false;

  //instantiates the specific Tracker
  Ptr<Tracker> tracker = createTrackerByName(tracker_algorithm);
  if (!tracker)
  {
    cout << "***Error in the instantiation of the tracker...***\n";
    return -1;
  }

  //get the first frame
  cap >> frame;
  frame.copyTo( image );
  if(initBoxWasGivenInCommandLine){
      boundingBox.x = coords[0];
      boundingBox.y = coords[1];
      boundingBox.width = std::abs( coords[2] - coords[0] );
      boundingBox.height = std::abs( coords[3]-coords[1]);
      printf("bounding box with vertices (%d,%d) and (%d,%d) was given in command line\n",coords[0],coords[1],coords[2],coords[3]);
      rectangle( image, boundingBox, Scalar( 255, 0, 0 ), 2, 1 );
  }
  else {
    video.open(outputFile, VideoWriter::fourcc('M','J','P','G'), 28, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));

    // Create a 4D blob from a frame.
    blobFromImage(frame, blob, 1/255.0, cvSize(inpWidth, inpHeight), Scalar(0,0,0), true, false);
    
    //Sets the input to the network
    net.setInput(blob);
    
    // Runs the forward pass to get output of the output layers
    vector<Mat> outs;
    net.forward(outs, getOutputsNames(net));
    
    // Remove the bounding boxes with low confidence
    boundingBox = postprocess(frame, outs);
    
    // Write the frame with the detection boxes
    Mat detectedFrame;
    frame.convertTo(detectedFrame, CV_8U);

    video.write(detectedFrame);

    detectedFrame.copyTo(image);
  }
  
  imshow( "Tracking API", image );

  bool initialized = false;
  int frameCounter = 0;
  int64 timeTotal = 0;
  double outputFPS = 0;
  double curTime = 0;

  string outputLabel;

  for ( ;; )
  {
    if( !paused )
    {
      if(initialized){
          cap >> frame;
          if(frame.empty()){
            break;
          }
          frame.copyTo( image );
      }

      if( !initialized )
      {
        //initializes the tracker
        if( !tracker->init( frame, boundingBox ) )
        {
          cout << "***Could not initialize tracker...***\n";
          return -1;
        }
        initialized = true;
      }
      else if( initialized )
      {
        int64 frameTime = getTickCount();
        //updates the tracker
        if( tracker->update( frame, boundingBox ) )
        {
          rectangle( image, boundingBox, Scalar( 255, 0, 0 ), 2, 1 );
        }
        frameTime = getTickCount() - frameTime;
        timeTotal += frameTime;
      }
      frameCounter++;
      outputFPS = (frameCounter * freq)/timeTotal;
      curTime = cap.get(CAP_PROP_POS_MSEC)/1000;

      outputLabel = format("Object Detection with YOLOv3 ----- Current Time: %.2f ----- FPS: %.2f", curTime, outputFPS);

      putText(image, outputLabel, Point(20,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);

      Mat trackedOutputFrame;
      image.convertTo(trackedOutputFrame, CV_8U);
      video.write(trackedOutputFrame);

      imshow( "Tracking API", image );
    }

    char c = (char) waitKey( 2 );
    if( c == 'q' )
    {
      cap.release();
      video.release();
      break;
    }
    if( c == 'p' )
      paused = !paused;
  }

  double s = frameCounter / (timeTotal / getTickFrequency());
  printf("FPS: %f\n", s);

  return 0;
}


// Remove the bounding boxes with low confidence using non-maxima suppression
Rect postprocess(Mat& frame, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
    if (boxes.size() > 0)
      return boxes[0];
    else
      return Rect(0,0,0,0);
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);
    
    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }
    
    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
}

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}