// -----------------------------------------------------------------------------
//
//  Gmsh GEO: Symmetry ASTM Standard E-399-72
//
// -----------------------------------------------------------------------------
//
// Symmetry ASTM Standard E-399-72
//
//
//gmsh file.geo  -3 -o mesh.msh
//
//   *-------------------------*
//   |    ---                  |
//   |  /     \                |
//   | |   *   |               |
//   |  \     /                |
//   |    ---                  |
//   |                         |
//    ----------------         |
//   |                         |
//   |    ---                  |
//   |  /     \                |
//   | |   *   |               |
//   |  \     /                |
//   |    ---                  |
//   *-------------------------*
//
//
//   *-------------------------*
//   |    ---                  |
//   |  /     \                |
//   | |   *   |               |
//   |  \     /                |
//   |    ---                  |
//   |                         |
//   *-------------------------*
//
//

h      = 0.05;  //mesh size
hcrack = 0.01; //mesh size near crack
SetFactory("OpenCASCADE");

// ------------------------------------------------------
// ------------------------------------------------------
// A)Geometry Definition: 1)Points 
//                        2)Lines 
//                        3)Curve 
//                        4)Surface 

// ------------------------------------------------------
// A1)Points Definitions: 
//              
//         P4*----------*P3
//           |          |
//           |          |
//         P1*----------*P2
//             
//    |Y
//    |
//    ---X  
// Z /
//

//           -----Coordinates--
//Points:    -----X,------Y,---Z,
Point(1)   ={ -0.25,    0.0,   0,  h};
Point(2)   ={   1.0,    0.0,   0,  h};
Point(3)   ={   1.0,    0.6,   0,  h};
Point(4)   ={ -0.25,    0.6,   0,  h};


// ------------------------------------------------------
// A2)Lines Definition
//
//            <-L3
//        *----------*
//     |L4|          |
//        |          | ^L2
//        |          |
//        *----------*
//           L1->

Line(1) = {1, 2};  //L1:from P1 to P2: P1*--L1-->*P2
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};


//        *----------------* 
//        |  / - \         |  
//        |  |   |         |
//        |  \ - /         |
//        *     -----------| 
//        |  / - \         |  
//        |  |   |         |
//        |  \ - /         |
//        *----------------*

Circle(39) = {0.0,  0.275, 0.0, 0.125, 0.0,   Pi};
Circle(40) = {0.0,  0.275, 0.0, 0.125,  Pi, 2*Pi};


// ------------------------------------------------------
// A3)Curve Definition
//
//            
//        *----<-----*
//        |          |
//        *  \       |
//            *      ^ Curve 5
//        *  /       | 
//        |          |
//        *----->----*
//   

Curve Loop(5) = {1,2,3,4};  //C5: through lines L1,L2,...,L7
Curve Loop(6) = {39, 40};  



// ------------------------------------------------------
// A4)Surface Definition
//
//         

Plane Surface(6) = {5, 6};  // Subtract circle loops 39 and 40 from the main surface 5
//Recombine Surface {6};


// ------------------------------------------------------
// ------------------------------------------------------
// B)Mesh Generation: 1)Mesh size Box1 
//                    2)Mesh size Box2
//                    3)Mesh min(Box1,Box2)
//                    3)Extrude Mesh 
//                    4)Mesh Algorithm  


// ------------------------------------------------------
// B1) Mesh size Box1
//
//        *----------------* 
//        |  / - \         |  
//        |  |   | (Field[6])  
//        |  \ - /         |
//         -----------     | 
//        |                |
//        |                |
//        |                |
//        *----------------*

Field[6]      = Ball;
Field[6].VIn  = hcrack;
Field[6].VOut = h;

Field[6].XCenter = 0.0;        // X-coordinate of the center of the circle
Field[6].YCenter = 0.275;    // Y-coordinate of the center of the circle
Field[6].ZCenter = 0.0;        // Z-coordinate of the center of the circle (for 2D, keep it zero)
Field[6].Radius  = 0.145;    // Radius of the circle


// ------------------------------------------------------
// B3) Mesh size Box2
//


Field[8]      =    Box;
Field[8].VIn  = hcrack;
Field[8].VOut =      h;

Field[8].XMin =  0.45;
Field[8].XMax =   1.0;
Field[8].YMin =   0.0;
Field[8].YMax =   0.1;

// ------------------------------------------------------
// B3) Mesh min(Box1,Box2)
Field[9] = Min;
Field[9].FieldsList = {6,8};
Background Field = 9;


// ------------------------------------------------------
// B4)Extrude Mesh

//     {X, Y,    Z}    Surface
Extrude{0, 0,  0.5}{Surface{6}; Layers{{2},{1}}; Recombine;}


// ------------------------------------------------------
// B5)Mesh Algorithm
Geometry.Tolerance = 1e-12;
Mesh.SaveAll = 1;

// ------------------------------------------------------
// Physical groups definition
//
Physical Volume("speciment", 67) = {1};
