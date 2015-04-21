#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <algorithm>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

Mat src, edges;

struct ellipse_data
{
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



vector<vector<vector<short> > > store_hiearchical_pyramid(Mat& edgesM, int min_size = 32)
{
    cout << edgesM.rows << " " << edgesM.cols << endl;
    int log2h = (int)floor(log2(edgesM.rows));
    int log2w = (int)floor(log2(edgesM.cols));
    int log2min = min(log2h, log2w);
    unsigned int max_size = pow(2, log2min);
    Mat resized = Mat(max_size,max_size,edgesM.type());
    resize(edgesM, resized, resized.size());

    unsigned int steps = log2min - (int)floor(log2(min_size)) + 1;
    vector<vector<vector<short> > > pyramid = vector<vector<vector<short> > >(steps);
    pyramid[0] = vector<vector<short> >(max_size,vector<short>(max_size,0));
    for (int i = 0; i < max_size; ++i) {
        for (int j = 0; j < max_size; ++j) {
            uchar p = resized.at<uchar>(i,j);
            if (p > 0)
                pyramid[0][i][j] = 1;
        }
    }

    unsigned int sz = max_size;
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

bool can_belong_to_ellipse(int x, int y, ellipse_data e, double eps = 0.1)
{
    double _x = cos(e.orient) * (x - e.x0) + sin(e.orient) * (y - e.y0);
    double _y = sin(e.orient) * (x - e.x0) - cos(e.orient) * (y - e.y0);
    double eqt = (_x * _x)/(e.a*e.a) + (_y * _y)/(e.b*e.b);
    return abs(eqt - 1) < eps;
}

vector<pair<ellipse_data,int> > hough_transform(vector<vector<short> > data, int min_vote = 1, int min_dist = 0)
{
    int xstart = 0;
    int xfin = (int)data.size();
    int ystart = 0;
    int yfin = (int)data[0].size();
    vector<pair<ellipse_data,int> > res;
    int maxlen = cvRound(sqrt(data.size()*data.size() + data[0].size()*data[0].size()));

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
                            int scaled_ma = cvRound(minor_axis);
                            //int ma_ceil = cvCeil(minor_axis);
                            //int ma_floor = cvFloor(minor_axis);
                            if (scaled_ma > min_dist) {
                                acc[scaled_ma] += data[y][x];
                                //acc[ma_ceil]++;
                                //acc[ma_floor]++;
                            }
                        }
                    }

                    int maxlength = 0;
                    int maxvote = 0;
                    for (int k = 0; k < maxlen; ++k) {
                        if (acc[k] > maxvote)
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
                        res.push_back(make_pair(ellipse_data(x0,y0, a, maxlength,orient, x1, y1, x2, y2),maxvote));
                    }
                }
            }
        }
    }

    return res;
}


bool is_ellipse_point(int x, int y, ellipse_data e, double eps = 0.1)
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

void write_ellipses_to_file(vector<pair<ellipse_data,int> > & ellipses, const char* fname = "data.txt")
{
    ofstream myfile;
    myfile.open (fname);
    for (int i = 0; i < ellipses.size(); i++)
    {
        myfile << ellipses[i].second;
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

ellipse_data relocate_ellipse(vector<vector<short> > data, ellipse_data prev, int min_vote = 1, int min_dist = 0)
{
    int xstart = min(2*prev.x1 - 1, 2*prev.x2 - 1);
    int xfin = max(2*prev.x1 + 1, 2*prev.x2 + 1);
    int ystart = min(2*prev.y1 - 1, 2*prev.y2 - 1);
    int yfin = max(2*prev.y1 + 1, 2*prev.y2 + 1);
    vector<pair<ellipse_data,int> > res;
    cout << data.size() << " x " << data[0].size() << endl;
    cout << xstart << " - " << xfin << endl;
    cout << ystart << " - " << yfin << endl;
    int maxlen = cvRound(sqrt(data.size()*data.size() + data[0].size()*data[0].size()));

    int acc[maxlen]; //really, MUCH less. TODO: Fix.
    for (int i = 0; i < maxlen; ++i) { //clear the accumulator array
        acc[i] = 0;
    }

    write_data_to_file(data);

    for (int y1 = 2 * prev.y1 - 1; y1 <= 2 * prev.y1 + 1; y1++)
    {
        for (int x1 = 2 * prev.x1 - 1; x1 <= 2 * prev.x1 + 1; x1++) //point (x1, y1) - side point of major axis
        {
            if (data[y1][x1] == 0)
                continue;

            //cout << x1 << " " << y1 << endl;

            for (int y2 = 2 * prev.y2 - 1; y2 <= 2 * prev.y2 + 1; y2++)
            {
                for (int x2 = 2 * prev.x2 - 1; x2 <= 2 * prev.x2 + 1; x2++) //point (x2, y2) - side point of major axis
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
                            int scaled_ma = cvRound(minor_axis);

                            if ((scaled_ma < 2*prev.b - 1) || (scaled_ma > 2*prev.b + 1))
                                continue;
                            //cout << scaled_ma << " , prev " << prev.b << endl;

                            if (scaled_ma > min_dist) {
                                acc[scaled_ma] += data[y][x];
                                //acc[ma_ceil]++;
                                //acc[ma_floor]++;
                            }
                        }
                    }

                    int maxlength = 0;
                    int maxvote = 0;
                    for (int k = 0; k < maxlen; ++k) {
                        if (acc[k] > maxvote)
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
                        //cout << maxvote << " " << maxlength << " " << a << " " << x1 << " " << y1 << " " << x2 << " " << y2 << " " << orient <<endl;
                        res.push_back(make_pair(ellipse_data(x0,y0, a, maxlength,orient, x1, y1, x2, y2),maxvote));
                    }
                }
            }
        }
    }
    cout << "After relocating left:" << res.size() << endl;
    res = remove_duplicates(data, res, 5);
    pair<ellipse_data, int> e = *max_element(res.begin(), res.end(), compr);
    //cout << e.second << endl;
    return e.first;
}


double ellipse_length(ellipse_data elp, int step = 0)
{
    int mult = pow(2,step);
    double a = elp.a * mult;
    double b = elp.b * mult;
    //return 4 * (CV_PI * a * b + pow(a - b, 2))/ (a+b);
    return CV_PI * (a+b);
}

int nmb_of_ellipses = 1;

ellipse_data ellipse_detection(Mat src, int min_vote = 90, int min_dist = 5)
{
    Canny(src, edges, 50, 200, 3);
    vector<vector<vector<short> > > pyramid = store_hiearchical_pyramid(edges, 64);
    int n = (int)pyramid.size();
    cout << n << endl;
    write_data_to_file(pyramid[n-1], "data1.txt");
    vector<pair<ellipse_data,int> > ellipses = hough_transform(pyramid[n - 1], 10, 2);
    cout << ellipses.size() << endl;
    write_ellipses_to_file(ellipses, "ellipsesvotes0.txt");
    //ellipses = remove_duplicates(pyramid[n - 1],ellipses);
    if (ellipses.size() < 1) {
        cout << "Didn't find anything" << endl;
        return ellipse_data(0, 0, -1, -1, 0);
    }
    write_ellipses_to_file(ellipses, "ellipsesvotes.txt");
    ellipse_data found = (*max_element(ellipses.begin(), ellipses.end(), compr2)).first;
    cout << "(" << found.x0 << " " << found.y0 << " " << found.a << " " << found.b << " " << found.orient << ") " << endl;
    cout << "It got " << (*max_element(ellipses.begin(), ellipses.end(), compr2)).second << " votes." << endl;
    cout << "Approximate length = " << ellipse_length(found,n-1) << endl;
    //Debug output
    Mat drawing_canvas = src.clone();
    draw_ellipse(drawing_canvas,found, n-1);
    for (int i = n-2; i >= 0; i--)
    {
        cout << i << endl;
        found = relocate_ellipse(pyramid[i],found);

        draw_ellipse(drawing_canvas,found, i);

        //TODO: не знаю, как, но иногда в процессе эллипс все же теряется
        cout << "(" << found.x0 << " " << found.y0 << " " << found.a << " " << found.b << " " << found.orient << ") " << endl;
    }
    imwrite("ellipses3.jpg", drawing_canvas);

    return found;
}

int main( int argc, char** argv ) {
    src = imread( argv[1], 1 );
    Canny(src, edges, 50, 200, 3);
    cout << edges.size() << endl;
    imwrite("Edges.jpg", edges);

    ellipse_data elp = ellipse_detection(src,5,5);
    cout << "Found: (" << elp.x0 << " " << elp.y0 << " " << elp.a << " " << elp.b << " " << elp.orient << ") " << endl;
    Mat ellipses_draw = src.clone();
    ellipse(ellipses_draw, Point_<int>(elp.x0,elp.y0), Size_<double>(elp.a,elp.b), elp.orient * 180/ CV_PI, 0, 360, Scalar(0,0,255), 1, LINE_AA);
    //imshow( "Ellipses", ellipses_draw );
    imwrite("ellipses2.jpg", ellipses_draw);



    return 0;
}



ellipse_data::ellipse_data(int _x0, int _y0, double _a, double _b, double th): x0(_x0), y0(_y0), a(_a), b(_b), orient(th){
}

ellipse_data::ellipse_data(int _x0, int _y0, double _a, double _b, double th, int _x1, int _y1, int _x2, int _y2): x0(_x0), y0(_y0), a(_a), b(_b), orient(th), x1(_x1), y1(_y1), x2(_x2), y2(_y2) {
}
