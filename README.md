This is the implementation of Hough transform using one-dimensional accumulator array and hierarchical pyramid.

Build:
> cmake .

> make

NOTE: Requires OpenCV (3.0 recommended).

Run:

> ./Ellipse_detection <input file> \[<output file>\]

For example:

> ./Ellipse_detection samples/s1.jpg

This should print the parametres of the ellipse and write the image with detected ellipse to ellipse.jpg.
