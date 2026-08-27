// -----------------------------------------------------------------------------
//
//  Gmsh GEO
//
// -----------------------------------------------------------------------------

//Use the following line to generate the mesh (.inp (abaqus))
//gmsh file.geo  -2 -o mesh.msh

h      = 0.02;  //mesh size
hcrack = 0.004; //mesh size near crack

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
//         P5*  \       |
//                *P7   *P8
//         P6*  /       |
//           |          |
//         P1*----------*P2
//             
//    |Y
//    |
//    ---X   Dimensions:   1 x 1 x 0.01
// Z /
//

//           -----Coordinates--
//Points:    ----X,------Y,---Z,
Point(1)   ={ -0.5,   -0.5,   0,  h};
Point(2)   ={  0.5,   -0.5,   0,  h};
Point(3)   ={  0.5,    0.5,   0,  h};
Point(4)   ={ -0.5,    0.5,   0,  h};
Point(5)   ={ -0.5,  0.001,   0,  h};
Point(6)   ={ -0.5, -0.001,   0,  h};
Point(7)   ={  0.0,    0.0,   0,  h};
Point(8)   ={  0.5,    0.0,   0,  h};


// ------------------------------------------------------
// A2)Lines Definition
//
//            <-L3
//        *----------*
//     |L4|          |
//        *  \L5     | ^
//            *      |L2
//        *  /L6     |
//      L7|          |
//        *----------*
//           L1->

Line(1) = {1, 2};  //L1:from P1 to P2: P1*--L1-->*P2
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 7};
Line(6) = {7, 6};
Line(7) = {6, 1};


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

Curve Loop(5) = {1,2,3,4,5,6,7};  //C5: through lines L1,L2,...,L7

// ------------------------------------------------------
// A4)Surface Definition
//
//        *----------*
//        |          |
//        *  \       |
//            *  S6  | 
//        *  /       | 
//        |          |
//        *----------*
//         

Plane Surface(6) = {5};  //  Curve loop 5 C5 --> Surface S6
Recombine Surface {6};


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
//        |                | 
//        |                |
//        |                |
//        *     -----------|
//             | hcrack F6 |  (Field[6])
//        *     -----------| 
//        |                |
//        |                |
//        |                |
//        *----------------*

Field[6]      =    Box;
Field[6].VIn  = hcrack;
Field[6].VOut =      h;

Field[6].XMin = -0.05;
Field[6].XMax =   0.5;
Field[6].YMin = -0.05;
Field[6].YMax =  0.05;


// ------------------------------------------------------
// B2) Mesh size Box2
//
//        *----------------*  
//        |                |
//        |                |
//        |   -------------|
//        |  |             |
//        *  |  hcrack*10  |
//        *  |   F7        | (Field[7])
//        |  |             |
//        |   -------------|
//        |                |
//        |                |
//        *----------------*

Field[7]      =       Box;
Field[7].VIn  = hcrack*10;
Field[7].VOut =         h;

Field[7].XMin =      -0.15;
Field[7].XMax =       0.5;
Field[7].YMin =      -0.2;
Field[7].YMax =       0.2;


// ------------------------------------------------------
// B3) Mesh min(Box1,Box2)
//
//        *----------------*  
//        |                |
//        |   -------------|
//        |  |  hcrack*10  |
//        *  |   ----------|
//        |  |  | hcrack F8| Field[8]
//        *  |   ----------| 
//        |  |           F8|
//        |   -------------|
//        |                |
//        *----------------*

Field[8] = Min;
Field[8].FieldsList = {6,7};
Background Field    = 8;



// ------------------------------------------------------
// B4)Extrude Mesh


// ------------------------------------------------------
// B5)Mesh Algorithm
// Geometry.Tolerance = 1e-12;
// Mesh.SaveAll = 1;

Geometry.Tolerance = 1e-12;
Mesh.Algorithm = 8;                    // Frontal-Delaunay for quads
Mesh.RecombineAll = 1;                 // Recombine all surfaces
Mesh.SubdivisionAlgorithm = 1;         // All quads subdivision
Mesh.RecombinationAlgorithm = 1;       // Simple recombination

// ------------------------------------------------------
// Physical groups definition
//
//              "top"
//           *----------*
//           |          |
//           *  \       |
//                *     *
//           *  /       |
//           |          |
//           *----------*
//             "bottom"
//

//+
Physical Curve("bottom", 8) = {1};
//+
Physical Curve("top", 9) = {3};
//+
Physical Surface("specimen", 10) = {6};
