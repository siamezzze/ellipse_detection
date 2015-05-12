#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <algorithm>
#include <iostream>
#include <fstream>

#define DEBUG

using namespace std;
using namespace cv;

Mat src;

struct ellipse_data
{

    ellipse_data(): x0(-1), y1(-1) {};
    ellipse_data(int _x0, int _y0, double _a, double _b, double th);
    ellipse_data(int _x0, int _y0, double _a, double _b, double th, int _x1, int _y1, int _x2, int _y2);

    double a, b;
    int x0, y0;
    int x1, y1, x2, y2;

    double orient;
};

inline double distance(int x1, int y1, int x2, int y2)
{
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

inline pair<int,int> compute_center(int x1, int y1, int x2, int y2)
{
    return make_pair(cvRound((x1 + x2)/2), cvRound((y1 + y2)/2));
}

inline double major_axis_length(int x1, int y1, int x2, int y2)
{
    return distance(x1, y1, x2, y2) / 2;
}

inline double orientation(int x1, int y1, int x2, int y2)
{
    if (x1 == x2)
        return CV_PI/2; //because I don't really know what to do here
    return atan((double)(y2 - y1)/(double)(x2-x1));
}



vector<vector<vector<short> > > store_hiearchical_pyramid(Mat& edgesM, int min_size = 32) //create hierarchical pyramid of minimized images
{
    #ifdef DEBUG
    cout << edgesM.rows << " " << edgesM.cols << endl;
    #endif
    int log2h = (int)floor(log2(edgesM.rows));
    int log2w = (int)floor(log2(edgesM.cols));
    int log2min = min(log2h, log2w);
    unsigned int max_size = pow(2, log2min);
    Mat resized = Mat(max_size,max_size,edgesM.type());
    resize(edgesM, resized, resized.size()); //resize to square

    unsigned int steps = (unsigned int)log2min - (unsigned int)floor(log2(min_size)) + 1;
    vector<vector<vector<short> > > pyramid = vector<vector<vector<short> > >(steps);
    pyramid[0] = vector<vector<short> >(max_size,vector<short>(max_size,0)); //creating the first image (original size)
    for (int i = 0; i < max_size; ++i) {
        for (int j = 0; j < max_size; ++j) {
            uchar p = resized.at<uchar>(i,j);
            if (p > 0)
                pyramid[0][i][j] = 1;
        }
    }

    unsigned int sz = max_size; //create next levels by summing pixels of previous ones.
    for (int i = 1; i < steps; i++)
    {
        sz /= 2;
        pyramid[i] = vector<vector<short> >(sz,vector<short>(sz,0));
        for (int j = 0; j < sz; j++)
        {
            for (int k = 0; k < sz; ++k) {
                pyramid[i][j][k] = pyramid[i-1][j*2][k*2] + pyramid[i-1][j*2+1][k*2] + pyramid[i-1][j*2][k*2+1] + pyramid[i-1][j*2+1][k*2+1];
            }
        }
    }
    return pyramid;
}

inline bool can_belong_to_ellipse(int x, int y, ellipse_data e, double eps = 0.1) //for cleaning the image
{
    double _x = cos(e.orient) * (x - e.x0) + sin(e.orient) * (y - e.y0);
    double _y = sin(e.orient) * (x - e.x0) - cos(e.orient) * (y - e.y0);
    double eqt = (_x * _x)/(e.a*e.a) + (_y * _y)/(e.b*e.b);
    return abs(eqt - 1) < eps;
}

#ifdef DEBUG
vector<Point2d> badpoints = vector<Point2d>();
#endif

//First search (should be made on the minimized image)
vector<pair<ellipse_data,int> > hough_transform(vector<vector<short> > data, int min_vote = 1, int min_dist = 0, int scale = 16)
{
    int xstart = 0;
    int xfin = (int)data.size();
    int ystart = 0;
    int yfin = (int)data[0].size();
    vector<pair<ellipse_data,int> > res;
    int maxlen = scale*cvRound(sqrt(data.size()*data.size() + data[0].size()*data[0].size()));

    int acc[maxlen];
    for (int i = 0; i < maxlen; ++i) { //clear the accumulator array
        acc[i] = 0;
    }

    for (int y1 = ystart; y1 < yfin; y1++)
    {
        for (int x1 = xstart; x1 < xfin; x1 ++) //point (x1, y1) - side point of major axis
        {
            if (data[y1][x1] == 0)
                continue;

            //cout << x1 << " " << y1 << endl;

            for (int y2 = y1; y2 < yfin; y2++)
            {
                for (int x2 = 0; x2 < xfin; x2 ++) //point (x2, y2) - side point of major axis
                {
                    if (((y2 == y1) && (x2 <= x1)) || (data[y2][x2] == 0))
                        continue;

                    double dist = distance(x1, y1, x2, y2);

                    if (dist < min_dist)
                        continue;


                    pair<int,int> center = compute_center(x1, y1, x2, y2);

                    int x0 = center.first;
                    int y0 = center.second;
                    double a = major_axis_length(x1, y1, x2, y2); //major axis (half)
                    double orient = orientation(x1, y1, x2, y2);

                    //cout << x1 << " " << y1 << " " << x2 << " " << y2 << endl;

                    for (int y = ystart; y < yfin; y++)
                    {
                        for (int x = xstart; x < xfin; x++) //point of the ellipse (x,y)
                        {
                            if (((x == x1) && (y == y1)) || ((x == x2) && (y == y2)) || (data[y][x] == 0))
                                continue;

                            double d = distance(x, y, x0, y0);

                            if ((d < min_dist) || (d > a))
                                continue;


                            double f = min(distance(x, y, x1, y1), distance(x, y, x2, y2));
                            double costh_sq = (a * a + d * d - f * f) / (2 * a * d);
                            if (costh_sq > 1) //well, that's certainly a mistake, but i don't know how to handle it correctly
                            {
                                costh_sq = 1;
                            }
                            double sinth_sq = 1 - costh_sq; //at least one obvious thing
                            double b_sq = (a * a * d * d * sinth_sq) / (a * a - d * d * costh_sq);

                            double minor_axis = sqrt(b_sq); //i hope i got it right

                            int scaled_ma = cvRound(scale*minor_axis);
                            int ma_ceil = cvCeil(scale*minor_axis);
                            int ma_floor = cvFloor(scale*minor_axis);
                            double part = ma_ceil - scale*minor_axis;

                            if (scaled_ma > min_dist) { //voting

                                #ifdef DEBUG
                                if ((x1 == 8) && (y1 == 4) && (x2 == 8) && (y2 == 16) && ((scaled_ma == 12.875 * scale))) {
                                    cout << "(" << x << "," << y << ") = " << data[y][x] << endl;
                                    badpoints.push_back(Point2d(x, y));

                                }
                                #endif
                                //acc[scaled_ma] += 1;
                                acc[scaled_ma] += 1;
                                //acc[ma_ceil]+=data[y][x] * (1 - part);
                                //acc[ma_ceil-1]+=data[y][x]/2;
                                //acc[ma_floor]+=data[y][x] * part;
                                //acc[ma_floor+1]+=data[y][x]/2;
                            }
                        }
                    }

                    int maxlength = 0;
                    int maxvote = 0;
                    for (int k = 0; k < maxlen; ++k) {
                        if (acc[k] >= maxvote)
                        {
                            maxvote = acc[k];
                            maxlength = k;
                        }
                        acc[k] = 0;
                    }
                    //cout << maxvote << " " << dist << endl;

                    if (maxvote > min_vote )
                    {
                        //we found something
                        //cout << maxvote << " " << maxlength << " " << a << " " << x1 << " " << y1 << " " << x2 << " " << y2 << " " << orient <<endl;
                        //cout << maxvote << ": " << y1 << " " << y2 << endl;
                        res.push_back(make_pair(ellipse_data(x0,y0, a, maxlength/(float)scale,orient, x1, y1, x2, y2),maxvote));
                    }
                }
            }
        }
    }

    return res;
}


inline bool is_ellipse_point(int x, int y, ellipse_data e, double eps = 0.1)
{
    double _x = cos(e.orient) * (x - e.x0) + sin(e.orient) * (y - e.y0);
    double _y = sin(e.orient) * (x - e.x0) - cos(e.orient) * (y - e.y0);
    double eqt = (_x * _x)/(e.a*e.a) + (_y * _y)/(e.b*e.b);
    return abs(eqt - 1) < eps;
}

bool compr(pair<ellipse_data,int> a, pair<ellipse_data,int> b)
{
    return a.second > b.second;
}

bool compr2(pair<ellipse_data,int> a, pair<ellipse_data,int> b)
{
    return a.second < b.second;
}



//Have no use for this one, though it is supposed to deal with duplucates nearby
vector<pair<ellipse_data,int> > remove_duplicates(vector<vector<short> > data, vector<pair<ellipse_data,int> > ellipses, int thr = 20)
{
    sort(ellipses.begin(), ellipses.end(), compr);
    vector<pair<ellipse_data,int> > res;
    for (int i = 0; i < ellipses.size(); i++)
    {
        res.push_back(pair<ellipse_data,int>(ellipses[i].first, 0));
    }

    for (int y = 0; y < data.size(); y++)
    {
        for (int x = 0; x < data[y].size(); x++)
        {
            if (data[y][x] == 0)
                continue;
            for (int i = 0; i < res.size(); i++)
            {
                if (is_ellipse_point(x,y,res[i].first))
                {
                    //cout << i << " " << x << " " << y << endl;
                    res[i].second+=data[x][y];
                    break;
                }
            }
        }
    }
    for (long i = res.size()-1; i >= 0; i--)
    {
        if (res[i].second <= thr)
            res.erase(res.begin()+i);
    }
    //sort(res.begin(), res.end(), compr);
    return res;
}


double ellipse_length(ellipse_data elp, int step = 0)
{
    int mult = pow(2,step);
    double a = elp.a * mult;
    double b = elp.b * mult;
    //return 4 * (CV_PI * a * b + pow(a - b, 2))/ (a+b);
    return CV_PI * (a+b);
}

bool compr3(pair<ellipse_data,int> a, pair<ellipse_data,int> b)
{
    return (a.second / ellipse_length(a.first)) < (b.second / ellipse_length(b.first));
}

//Debug output

//image data to text file
void write_data_to_file(vector<vector<short> > & data, const char* fname = "data.txt")
{
    ofstream myfile;
    myfile.open (fname);
    for (int i = 0; i < data.size(); i++)
    {
        for (int j = 0; j < data[0].size(); j++)
        {
            myfile << data[i][j] << " ";
        }
        myfile << endl;
    }
    myfile.close();
}

void write_ellipses_to_file(vector<pair<ellipse_data,int> > & ellipses, const char* fname = "data.txt", int step = 0)
{
    ofstream myfile;
    myfile.open (fname);
    for (int i = 0; i < ellipses.size(); i++)
    {
        ellipse_data found = ellipses[i].first;
        myfile << "(" << found.x0 << " " << found.y0 << " " << found.a << " " << found.b << " " << found.orient << ") [" << found.x1 << ", " << found.y1 << ", " << found.x2 << ", " <<found.y2<< "] " << endl;
        myfile << ellipses[i].second << " " << ellipses[i].second / ellipse_length(found,step) << endl;
        myfile << endl;
    }
    myfile.close();
}

//draw ellipse
void draw_ellipse(Mat & canvas, ellipse_data elp, int step = 0)
{
    Scalar colors[5] = {Scalar(0,0,255),Scalar(0,255,0), Scalar(255,0,0), Scalar(0,100,200), Scalar(255,255,0)};
    //int multp = step + 1;
    ellipse(canvas, Point_<int>(elp.x0 * pow(2,step), elp.y0 * pow(2,step)), Size_<double>(elp.a * pow(2,step),elp.b *pow(2,step)), elp.orient * 180/ CV_PI, 0, 360, colors[step], 2, LINE_AA);
}

void draw_points(Mat & canvas, int step = 0)
{
    #ifdef DEBUG
    for (int i = 0; i < badpoints.size(); i++)
    {
        Point a = Point(badpoints[i].x*pow(2,step),badpoints[i].y*pow(2,step));
        Point b = Point(badpoints[i].x*pow(2,step) + 2,badpoints[i].y*pow(2,step));
        line(canvas, a, b, Scalar(0,0,0),5);
    }
    #endif
}

//end of debug output functions

ellipse_data relocate_ellipse(vector<vector<short> > data, ellipse_data prev, int scale = 16, int min_vote = 1, int min_dist = 0)
{
    //The area where we seek ellipse points (basically, all the image)
    int xstart = 0;
    int xfin = (int)data[0].size() - 1;
    int ystart = 0;
    int yfin = (int)data.size() - 1;
    vector<pair<ellipse_data,int> > res;
    #ifdef DEBUG
    cout << data.size() << " x " << data[0].size() << endl;
    #endif

    //scale means how minimized is the image and therefore how we should scale the free parameter (experimental)
    scale *= 10;
    int maxlen = scale*cvRound(sqrt(data.size()*data.size() + data[0].size()*data[0].size()));

    int acc[maxlen]; //really, MUCH less. TODO: Fix.
    for (int i = 0; i < maxlen; ++i) { //clear the accumulator array
        acc[i] = 0;
    }

    #ifdef DEBUG
    write_data_to_file(data);
    #endif

    //This time, we search for side points only as far as +- pixel from previously found ones
    int sidex1_start = max(xstart, 2 * prev.x1 - 2);
    int sidex1_fin = min(xfin, 2 * prev.x1 + 2);
    int sidex2_start = max(xstart, 2 * prev.x2 - 2);
    int sidex2_fin = min(xfin, 2 * prev.x2 + 2);
    int sidey1_start = max(ystart, 2 * prev.y1 - 2);
    int sidey1_fin = min(yfin, 2 * prev.y1 + 2);
    int sidey2_start = max(ystart, 2 * prev.y2 - 2);
    int sidey2_fin = min(yfin, 2 * prev.y2 + 2);

    for (int y1 = sidey1_start; y1 <= sidey1_fin; y1++)
    {
        for (int x1 = sidex1_start; x1 <= sidex1_fin; x1++) //point (x1, y1) - side point of major axis
        {
            if (data[y1][x1] == 0)
                continue;

            //cout << x1 << " " << y1 << endl;

            for (int y2 = sidey2_start; y2 <= sidey2_fin; y2++)
            {
                for (int x2 = sidex2_start; x2 <= sidex2_fin; x2++) //point (x2, y2) - side point of major axis
                {
                    if (((y2 == y1) && x2 <= x1) || (data[y2][x2] == 0))
                        continue;

                    double dist = distance(x1, y1, x2, y2);

                    if (dist < min_dist)
                        continue;


                    pair<int,int> center = compute_center(x1, y1, x2, y2);

                    int x0 = center.first;
                    int y0 = center.second;
                    double a = major_axis_length(x1, y1, x2, y2); //major axis (half)
                    double orient = orientation(x1, y1, x2, y2);

                    //cout << x1 << " " << y1 << " " << x2 << " " << y2 << endl;

                    for (int y = ystart; y < yfin; y++)
                    {
                        for (int x = xstart; x < xfin; x++) //point of the ellipse (x,y)
                        {
                            if (((x == x1) && (y == y1)) || ((x == x2) && (y == y2)) || (data[y][x] == 0))
                                continue;

                            double d = distance(x, y, x0, y0);

                            if ((d < min_dist) || (d > a))
                                continue;

                            double f = min(distance(x, y, x1, y1), distance(x, y, x2, y2));
                            double costh_sq = (a * a + d * d - f * f) / (2 * a * d);
                            if (costh_sq > 1) //well, that's certainly a mistake, but i don't know how to handle it correctly
                            {
                                costh_sq = 1;
                            }
                            double sinth_sq = 1 - costh_sq; //at least one obvious thing
                            double b_sq = (a * a * d * d * sinth_sq) / (a * a - d * d * costh_sq);

                            double minor_axis = sqrt(b_sq); //i hope i got it right
                            int scaled_ma = cvRound(minor_axis);

                            //if ((scaled_ma < 2*prev.b - 1) || (scaled_ma > 2*prev.b + 1))
                            //    continue;

                            scaled_ma = cvRound(scale*minor_axis);
                            int ma_ceil = cvCeil(scale*minor_axis);
                            int ma_floor = cvFloor(scale*minor_axis);
                            double part = ma_ceil - scale*minor_axis;

                            if (scaled_ma > min_dist) {
                                //acc[scaled_ma] += 1;
                                //acc[scaled_ma] += 1;
                                acc[ma_ceil]+=cvCeil(data[y][x] * (1 - part));
                                //acc[ma_ceil-1]+=data[y][x]/2;
                                acc[ma_floor]+=cvCeil(data[y][x] * part);
                                //acc[ma_floor+1]+=data[y][x]/2;
                            }
                        }
                    }

                    int maxlength = 0;
                    int maxvote = 0;
                    for (int k = 0; k < maxlen; ++k) {
                        if (acc[k] >= maxvote)
                        {
                            maxvote = acc[k];
                            maxlength = k;
                        }
                        acc[k] = 0;
                    }
                    //cout << maxvote << " " << dist << endl;

                    if ((maxvote > min_vote)  )
                    {
                        //we found something
                        //cout << maxvote << " " << maxlength << " " << a << " [" << x1 << " " << y1 << " " << x2 << " " << y2 << "] " << orient <<endl;
                        //cout << x1;
                        res.push_back(make_pair(ellipse_data(x0,y0, a, maxlength/(float)scale,orient, x1, y1, x2, y2),maxvote));
                    }
                }
            }
        }
    }
    #ifdef DEBUG
    cout << "After relocating left:" << res.size() << endl;
    #endif
    //res = remove_duplicates(data, res, 5);
    pair<ellipse_data, int> e = *max_element(res.begin(), res.end(), compr2);
    #ifdef DEBUG
    cout << "Votes" << e.second << endl;
    #endif
    return e.first;
}




void clear_picture(Mat& edges, ellipse_data elp)
{
    for (int x = 0; x < edges.cols; x++)
        for (int y = 0; y < edges.rows; y++)
        {
            if (can_belong_to_ellipse(x, y, elp, 1))
            {
                edges.at<uchar>(y,x) = 0;
            }
        }
}



ellipse_data ellipse_detection(Mat edges, int min_vote = 90, int min_dist = 5)
{

    Mat drawing_canvas = src.clone();

    vector<vector<vector<short> > > pyramid = store_hiearchical_pyramid(edges, 64);
    int n = (int)pyramid.size();
    #ifdef DEBUG
    cout << "Steps: " << n << endl;
    write_data_to_file(pyramid[n-1], "data1.txt");
    #endif
    vector<pair<ellipse_data,int> > ellipses = hough_transform(pyramid[n - 1], 4, 5, pow(2,n-1));
    #ifdef DEBUG
    cout << "Initial transform detected ellipses: " << ellipses.size() << endl;
    write_ellipses_to_file(ellipses, "ellipsesvotes0.txt");
    draw_points(drawing_canvas,n-1);
    #endif

    //some preparations can be made here
    //ellipses = remove_duplicates(pyramid[n - 1],ellipses);
    if (ellipses.size() < 1) {
        cout << "Didn't find anything" << endl;
        return ellipse_data(-1, -1, -1, -1, 0);
    }
    ellipse_data found = (*max_element(ellipses.begin(), ellipses.end(), compr2)).first;
    #ifdef DEBUG
    cout << "(" << found.x0 << " " << found.y0 << " " << found.a << " " << found.b << " " << found.orient << ") [" << found.x1 << ", " << found.y1 << ", " << found.x2 << ", " <<found.y2<< "] " << endl;
    cout << "It got " << (*max_element(ellipses.begin(), ellipses.end(), compr2)).second << " votes." << endl;
    cout << "Approximate length = " << ellipse_length(found,n-1) << endl;
    //Debug output
    draw_ellipse(drawing_canvas,found, n-1);
    #endif
    for (int i = n-2; i >= 0; i--)
    {
        cout << i << endl;
        found = relocate_ellipse(pyramid[i],found, pow(2,i));

        draw_ellipse(drawing_canvas,found, i);

        //TODO: не знаю, как, но иногда в процессе эллипс все же теряется
        #ifdef DEBUG
        cout << "(" << found.x0 << " " << found.y0 << " " << found.a << " " << found.b << " " << found.orient << ") [" << found.x1 << ", " << found.y1 << ", " << found.x2 << ", " <<found.y2<< "] " << endl;
        #endif
    }
    #ifdef DEBUG
    imwrite("ellipses3.jpg", drawing_canvas);
    #endif

    return found;
}

vector<ellipse_data> detect_ellipses(Mat src, int number_of_ellipses = 1)
{
    vector<ellipse_data> ellipses = vector<ellipse_data>(number_of_ellipses);
    Mat edges;
    Canny(src, edges, 50, 200, 3);
    //Probably some other transformation should be performed here, such as blurring
    #ifdef DEBUG
    imwrite("Edges.jpg", edges);
    #endif
    for (int i = 0; i < number_of_ellipses; i++)
    {
        ellipse_data elp = ellipse_detection(edges,5,5);
        ellipses[i] = elp;
        cout << "Found: (" << elp.x0 << " " << elp.y0 << " " << elp.a << " " << elp.b << " " << elp.orient << ") " << endl;
        clear_picture(edges, elp);
        #ifdef DEBUG
        imwrite("edges2.jpg", edges);
        #endif
    }
    return ellipses;

}

int nmb_of_ellipses = 2;

int main( int argc, char** argv ) {
    src = imread( argv[1], 1 );
    if(! src.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    Mat edges;

    Mat ellipses_draw = src.clone();
    vector<ellipse_data> ellipses = detect_ellipses(src, nmb_of_ellipses);
    for (int i = 0; i < ellipses.size(); i++)
    {
        ellipse_data elp = ellipses[i];
        if ((elp.x0 >= 0) && (elp.y0 >=0))
        {
            ellipse(ellipses_draw, Point_<int>(elp.x0,elp.y0), Size_<double>(elp.a,elp.b), elp.orient * 180/ CV_PI, 0, 360, Scalar(0,0,255), 1, LINE_AA);
        }
    }
    //imshow( "Ellipses", ellipses_draw );
    imwrite("ellipses2.jpg", ellipses_draw);



    return 0;
}



ellipse_data::ellipse_data(int _x0, int _y0, double _a, double _b, double th): x0(_x0), y0(_y0), a(_a), b(_b), orient(th){
}

ellipse_data::ellipse_data(int _x0, int _y0, double _a, double _b, double th, int _x1, int _y1, int _x2, int _y2): x0(_x0), y0(_y0), a(_a), b(_b), orient(th), x1(_x1), y1(_y1), x2(_x2), y2(_y2) {
}
