import numpy as np

def write_asc(path, vertices):
    '''
    Args:
        vertices: shape = (nver, 3)
    '''
    if path.split('.')[-1] == 'asc':
        np.savetxt(path, vertices)
    else:
        np.savetxt(path + '.asc', vertices)


def write_obj(obj_name, vertices, colors, triangles):
    ''' Save 3D face model
    Args:
        obj_name: 
        vertices: shape = (nver, 3)
        colors: shape = (nver, 3)
        triangles: shape = (ntri, 3)
    '''
    triangles = triangles.copy()
    triangles += 1 # meshlab start with 1
    
    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj + '.obj'
        
    # write obj
    with open(obj_name, 'wb') as f:
        
        # write vertices & colors
        for i in range(vertices.shape[0]):
            # s = 'v {} {} {} \n'.format(vertices[0,i], vertices[1,i], vertices[2,i])
            s = 'v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2])
            f.write(s)

        # write f: ver ind/ uv ind
        [k, ntri] = triangles.shape
        for i in range(triangles.shape[0]):
            # s = 'f {}/{} {}/{} {}/{} \n'.format(triangles[i, 0], triangles[i, 0], triangles[i, 1], triangles[i, 1], triangles[i, 2], triangles[i, 2])
            s = 'f {} {} {}\n'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2])
            f.write(s)

