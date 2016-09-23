# Super-Saiyan

Design
-----------------------------------------------------------------------------------------------------------------------------------------
Uses the OpenCV library's cascade classifier function to determine the region of interest containing the user's face.
A mask image of Goku's super saiyan hair is resized and super imposed (matrix addition of pixels) onto the current source image.
The source image may be fetched from a local directory or directly from the camera feed. 

Results
-----------------------------------------------------------------------------------------------------------------------------------------
Works well enough.
Current trade off between lag and flickering.
If enough movement is detected the mask image is resized and redrawn, otherwise it stays put (assuming the user's face didn't move much)

Hindsight
-----------------------------------------------------------------------------------------------------------------------------------------
There are some problems with OpenCV's functions interact with the Visual Studio environment, the program has errors upon exiting.
The functions also return really undesirable values that will make the program crash, so lots of conditional checks are required, it's 
not really worth debugging.
Recreate the program on a different platform (linux) and maybe just use a python script.
