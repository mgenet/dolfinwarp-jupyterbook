SetFactory("OpenCASCADE");


Mesh.Algorithm = 4;
Mesh.CharacteristicLengthMin = 0.05;
Mesh.CharacteristicLengthMax = 0.1;


Rectangle(1) = {0, 0, 0, 1, 1, 0};
Circle(4) = {0.5, 0.5, 0, 0.2};

//Physical Surface(1) = {1000};
