// -----------------------------------------------------------------------------
//
//  Gmsh GEO
//
//  Mesh size fields
//
// -----------------------------------------------------------------------------

//Use the following line to generate the mesh
//gmsh file.geo  -2 -o mesh.msh

// Parameters
Lx = 10;  // Length of the rectangle in x-direction
Ly = 5;   // Length of the rectangle in y-direction
ndiv_x = 10; // Number of divisions along the x-direction
ndiv_y = 5;  // Number of divisions along the y-direction


// ------------------------------------------------------
// ------------------------------------------------------
// A)Geometry Definition: 1)Points 
//                        2)Lines 
//                        3)Curve 
//                        4)Surface 

// ------------------------------------------------------
// A1)Points Definitions:                                  
//                                                     
//           4*-------------------------------*3   -
//            |                               |    |
//            |                               |    |
//            |               * (0,0)         |    |    Ly
//            |                               |    |
//            |                               |    |
//            *-------------------------------*    -
//            1                               2
//    |Y      |---------------Lx--------------|        
//    |                
//    ---X         


//          -----Coordinates--
//Points:   ----X,------Y,---Z,
Point(1) = {-Lx/2,  -Ly/2,   0, 1.0};
Point(2) = { Lx/2,  -Ly/2,   0, 1.0};
Point(3) = { Lx/2,   Ly/2,   0, 1.0};
Point(4) = {-Lx/2,   Ly/2,   0, 1.0};

// ------------------------------------------------------
// A2)Lines Definition
//
//                            <-L3
//            *-------------------------------* 
//            |                               | 
//            |                               | ^
//          L4|                               | L2
//          \/|                               |
//            |                               | 
//            *-------------------------------*
//                  L1->          

Line(1) = {1, 2};  // Bottom line
Line(2) = {2, 3};  // Right line
Line(3) = {3, 4};  // Top line
Line(4) = {4, 1};  // Left line


// ------------------------------------------------------
// A3)Curve Definition
//                            
//            *---------------<---------------* 
//            |                               | 
//            |                               | 
//           \/                               /\ 
//            |                               |
//            |                               | 
//            *--------------->---------------*
//                              

Line Loop(1) = {1, 2, 3, 4};





// ------------------------------------------------------
// A4)Surface Definition
//                           
//            *-------------------------------* 
//            |                               | 
//            |                               | 
//            |      S1                       | 
//            |                               |
//            |                               | 
//            *-------------------------------*
//                  

Plane Surface(1) = {1};

// Transfinite Surface and Recombination
Transfinite Surface {1};
Recombine Surface {1};

// Set transfinite lines for structured mesh with different divisions
Transfinite Line {1} = ndiv_x + 1;
Transfinite Line {2} = ndiv_y + 1;
Transfinite Line {3} = ndiv_x + 1;
Transfinite Line {4} = ndiv_y + 1;


// ------------------------------------------------------
// B4)Extrude Mesh (for 3D)

//      {X, Y,    Z} Surface
// Extrude {0, 0, 5.0}{Surface{1}; Layers{1};Recombine;}

// ------------------------------------------------------
// B5)Mesh Algorithm
Geometry.Tolerance = 1e-12;
