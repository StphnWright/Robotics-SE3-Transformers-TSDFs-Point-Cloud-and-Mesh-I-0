import numpy as np
import os

class Ply(object):
    """Class to represent a ply in memory, read plys, and write plys.
    """

    def __init__(self, ply_path=None, triangles=None, points=None, normals=None, colors=None):
        """Initialize the in memory ply representation.

        Args:
            ply_path (str, optional): Path to .ply file to read (note only
                supports text mode, not binary mode). Defaults to None.
            triangles (numpy.array [k, 3], optional): each row is a list of point indices used to
                render triangles. Defaults to None.
            points (numpy.array [n, 3], optional): each row represents a 3D point. Defaults to None.
            normals (numpy.array [n, 3], optional): each row represents the normal vector for the
                corresponding 3D point. Defaults to None.
            colors (numpy.array [n, 3], optional): each row represents the color of the
                corresponding 3D point. Defaults to None.
        """
        super().__init__()
        
        # If ply path is None, load in triangles, point, normals, colors.
        # else load ply from file. If ply_path is specified AND other inputs
        # are specified as well, ignore other inputs.
        if ply_path is not None:
            if not os.path.isfile(ply_path):
                raise ValueError('Invalid path')
            self.read(ply_path)
            return
        
        # Triangles
        if triangles is not None and (len(triangles.shape) != 2 or triangles.shape[1] != 3):
            raise ValueError('Invalid triangles list')
        self.triangles = triangles
        self.triangles_name = "face"
        self.triangles_prop_name = "vertex_index"
                
        # Points
        if points is not None and (len(points.shape) != 2 or points.shape[1] != 3):
            raise ValueError('Invalid points list')
        self.points = points
        self.points_name = "vertex"
        self.points_prop_names = ["x", "y", "z"]
        n = points.shape[0]
        
        # Normals
        # If normals are not None make sure that there are equal number of points and normals.
        if normals is not None:
            if len(normals.shape) != 2 or normals.shape[1] != 3:
                raise ValueError('Invalid colors list')
            if normals.shape[0] != n:
                raise ValueError('The number of colors does not match the number of points')
            self.normals_prop_names = ["nx", "ny", "nz"]
        self.normals = normals
        
        # If Colors are not None make sure that there are equal number of colors and normals.
        if colors is not None:
            if len(colors.shape) != 2 or colors.shape[1] != 3:
                raise ValueError('Invalid colors list')
            if colors.shape[0] != n:
                raise ValueError('The number of colors does not match the number of points')    
            self.colors_prop_names = ["red", "green", "blue"]
        self.colors = colors

        # Default precision for points and normals
        self.d = 3

    def read(self, ply_path):
        """Read a ply into memory.

        Args:
            ply_path (str): ply to read in.
        """
        # Read in ply.
        
        # Open the file
        with open(ply_path) as f:

            # First line should say "ply"
            line = f.readline().rstrip()
            if line != "ply":
                raise ValueError("Header does not start with the word 'ply'")
            
            # Check the format line
            line = f.readline().rstrip()
            if not line or not line.startswith("format"): 
                raise ValueError("Invalid format line")
            file_format = line.split(maxsplit = 3)
            if len(file_format) < 3:
                raise ValueError("Invalid format line")
            if file_format[1] != "ascii":
                raise ValueError("File is not ASCII-formatted")
            
            # Go to first "element" line (ignore other lines in between, if any)
            while not line.startswith("element"):
                line = f.readline().rstrip()
                if not line: 
                    raise ValueError("File header contains no elements")
            
            # Parse the element line
            elem = line.split(maxsplit=3)
            self.points_name = elem[1]
            n = int(elem[2])
            self.points = np.zeros((n, 3))
            
            # Read properties for the first element
            num_float_props = 0
            num_uchar_props = 0
            self.points_prop_names = []
            self.normals_prop_names = []
            self.colors_prop_names = []
            
            line = f.readline().rstrip()
            while line and (not line.startswith("element")) and (line != "end_header"):
                if line.startswith("property"):
                    prop = line.split(maxsplit = 3)
                    if (prop[1] == "float"):
                        # Floating point property
                        num_float_props += 1
                        if (num_float_props <= 3):
                            # Point
                            self.points_prop_names.append(prop[2])
                        elif (num_float_props <= 6):
                            # Normal
                            self.normals_prop_names.append(prop[2])
                    elif (prop[1] == "uchar"):
                        # uchar property (color)
                        num_uchar_props += 1
                        if (num_uchar_props <= 3):
                            self.colors_prop_names.append(prop[2])
                    
                # Next line
                line = f.readline().rstrip()
            
            # Check that there is a valid next line
            if not line: ValueError("File is missing end_header")
            
            # Validate number of float properties
            if num_float_props == 3:
                self.normals = None
                self.normals_prop_names = []
            elif num_float_props == 6:
                self.normals = np.zeros((n, 3))
            else:
                raise ValueError("Invalid number of point properties")
            
            # Validate number of uchar properties
            if num_uchar_props == 0:
                self.colors = None
                self.colors_prop_names = []
            elif num_uchar_props == 3:
                self.colors = np.zeros((n, 3))
            else:
                raise ValueError("Invalid number of point properties")
            
            # Default triangles structure
            self.triangles = None
            self.triangles_name = "face"
            self.triangles_prop_name = "vertex_index"
            
            # Check for faces
            if line.startswith("element"):
                
                # Parse the element
                elem = line.split(maxsplit = 3)
                self.triangles_name = elem[1]
                self.triangles = np.zeros((int(elem[2]), 3))
                
                # Next line should be a list property
                line = f.readline().rstrip()
                if not line or not line.startswith("property"):
                    raise ValueError("No properties found for faces")
                prop = line.split(maxsplit = 5)
                if prop[1] != "list" or prop[2] != "uchar" or prop[3] != "int":
                    raise ValueError("Invalid faces property")
                self.triangles_prop_name = prop[4]
            
            # Skip to end_header
            while line != "end_header":
                line = f.readline().rstrip()
                if not line: raise ValueError("File is missing end_header")
                
            # Process points
            num_expected_props = num_float_props + num_uchar_props
            for p in range(n):
                # Read the next point
                line = f.readline().rstrip()
                if not line: raise ValueError("File is missing points")
                
                # Split the line and check that the number of values is correct
                p_list = line.split()   
                if len(p_list) != num_expected_props:
                    raise ValueError("Incorrect number of point values")
                
                # Assign elements
                self.points[p, :] = p_list[0:3]
                if self.normals is not None:
                    self.normals[p, :] = p_list[3:6]
                if self.colors is not None:
                    self.colors[p, :] = p_list[-3:]
            
            # Determine the precision to use
            strNum = p_list[0]
            if "." not in strNum:
                self.d = 0
            else:
                n_list = strNum.split(".", maxsplit = 2)
                self.d = len(n_list[1])

            # Convert colors to int
            if self.colors is not None:
                self.colors = self.colors.astype(int)
                
            # Process faces
            if self.triangles is not None:
                for p in range(self.triangles.shape[0]):
                    # Read the next face
                    line = f.readline().rstrip()
                    if not line: raise ValueError("File is missing faces")

                    # Split the line and check that the number of values is correct
                    f_list = line.split()
                    if len(f_list) != 4 or int(f_list[0]) != 3:
                        raise ValueError("Invalid triangle in faces")

                    # Assign elements
                    self.triangles[p, :] = f_list[1:]
                
                # Convert faces to int
                self.triangles = self.triangles.astype(int)
                

    def write(self, ply_path):
        """Write mesh, point cloud, or oriented point cloud to ply file.

        Args:
            ply_path (str): Output ply path.
        """
        # Number of points and triangles
        num_points = self.points.shape[0]
        if self.triangles is not None:
            num_triangles = self.triangles.shape[0]
        
        # Open file for writing
        with open(ply_path, 'w') as f:
            # Write header depending on existance of normals, colors, and triangles
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element {self.points_name} {num_points:d}\n")
            for v in self.points_prop_names:
                f.write(f"property float {v}\n")
            for v in self.normals_prop_names:
                f.write(f"property float {v}\n")
            for v in self.colors_prop_names:
                f.write(f"property uchar {v}\n")
            if self.triangles is not None:
                f.write(f"element {self.triangles_name} {num_triangles:d}\n")
                f.write(f"property list uchar int {self.triangles_prop_name}\n")
            f.write("end_header\n")
            
            # Write points, write normals if they exist, and write colors if they exist
            for p in range(num_points):
                f.write(f"{self.points[p,0]:.{self.d:d}f} {self.points[p,1]:.{self.d:d}f} {self.points[p,2]:.{self.d:d}f}")
                if self.normals is not None:
                    f.write(f" {self.normals[p,0]:.{self.d:d}f} {self.normals[p,1]:.{self.d:d}f} {self.normals[p,2]:.{self.d:d}f}")
                if self.colors is not None:
                    f.write(f" {self.colors[p,0]:d} {self.colors[p,1]:d} {self.colors[p,2]:d}")
                f.write("\n")
                
            # Write face list if needed
            if self.triangles is not None:
                for p in range(num_triangles):
                    f.write(f"3 {self.triangles[p,0]:d} {self.triangles[p,1]:d} {self.triangles[p,2]:d}\n") 