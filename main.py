import os
import json
import time
import ast
import pymysql.cursors
import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint
from scipy.spatial.distance import cdist, cosine
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors

from config import *


#
INTERPOLATION_POINTS = 1000

def get_db_connection():
    '''
    Sets up the database connection
    '''
    connection = pymysql.connect(host=DB_HOST,
                                 user=DB_USERNAME,
                                 password=DB_PASSWORD,
                                 db=DB_DATABASE,
                                 cursorclass=pymysql.cursors.DictCursor)
    return connection


def read_material_json(cod_id):
    '''
    Reads material DOS information from json file
    '''
    material_path =  os.path.join(MATERIALS_PATH, str(cod_id), DOS_PATH)
    # Not all materials have a DOS json file
    try:
        data = json.load(open(material_path))
        # Not all DOS json files have DOS information
        if "dos" in data:
            return data
        else:
            #print("Error: this material doesn't have DOS information")
            return False

    except Exception as e:
        pass


def get_interpolation_window(connection, material):
    '''
    Gets the hvb value in the database and returns [hvb-2, hvb]
    '''
    with connection.cursor() as cursor:
            # Read a single record
            sql = (
                'SELECT DFT.HVB_E '
                'FROM materials_dft DFT, materials M '
                'WHERE M.material_id = DFT.material_id AND '
                'M._cod_database_code = ' + str(material)
            )

            # Parse result and return
            cursor.execute(sql)
            result = cursor.fetchone()
            hvb_e = result["HVB_E"]
            return [hvb_e-2., hvb_e]


def interpolate_dos(energy_window, e, dos):
    '''
    Returns an interpolation of (e, dos) in the 
    range energy_window made of INTERPOLATION_POINTS points.
    '''
    space = np.linspace(energy_window[0], energy_window[1], num=INTERPOLATION_POINTS)
    return np.interp(space, e, dos)


def get_all_material_ids(connection):
    '''
    Returns a list with all COD ID:s from the database
    '''
    with connection.cursor() as cursor:
        # Read a single record
        sql = (
            'SELECT M._cod_database_code '
            'FROM materials M '
        )
        # Parse result and return
        cursor.execute(sql)
        results = cursor.fetchall()
        return [material["_cod_database_code"] for material in results]


def get_interpolated_dos(connection, material_id):
    '''
    Takes a material id and return a DOS interpolation
    - Gets the material's json
    - Fetches the hvb_e in the database and computes the energy window
    - Uses the json E and dos to make an interpolation of the DOS in the window
    - Returns the DOS in this window
    '''
    material = read_material_json(material_id)
    if material:
        try:
            window = get_interpolation_window(connection, material_id)
            interpolated_dos = interpolate_dos(window, material["E"], material["dos"])
            return interpolated_dos

        except Exception as e:
            #print(e)
            raise(e)
    else:
        return False


def interpolate_all_dos(connection, all_materials):
    '''
    Takes a list of material ids and returns their DOS interpolation
    '''
    # Start timer
    now = time.time()

    # Init
    all_dos = []
    k = 0
    for _cod_database_code in all_materials:

        # Print progress
        if k % 1000 == 0:
            print("Interpolating DOS for all materials.. (" + str(k) 
            + "/" + str(len(all_materials)) + ")")
        k = k + 1

        # Handle errors (not all materials have valid DOS information)
        try:
            # Get the interpolated DOS for one material, append to array if succeeded
            dos = get_interpolated_dos(connection, _cod_database_code)
            if dos is not False:
                all_dos.append([_cod_database_code, dos])

        except Exception as e:
            print(e)

    # Print time and return
    later = time.time()
    difference = round(later - now, 3)
    print("Found " + str(len(all_dos)) + "/" + str(len(all_materials)) 
    + " materials with valid DOS information")
    print("Fetched and interpolated all DOS in " + str(difference) + " second(s)")
    return all_dos


def find_similar_materials(connection, all_dos, current_material_id):
    '''
    Computes euclidean distances for all materials to the reference
    and returns them as a list
    '''
    # Get all DOS for comparison
    other_dos = [dos[1] for dos in all_dos]
    other_ids = [dos[0] for dos in all_dos]

    # Get the reference DOS (there might be a faster/cleaner way to get it)
    base_dos = [x[1] for i,x in enumerate(all_dos) if x[0] == current_material_id][0]

    # Compute all the distances
    distances = cdist([base_dos], other_dos, metric = 'euclidean')[0]#).tolist()

    # Zip and return
    return list(zip(other_ids, distances))


def compute_distances():
    '''
    High-level method with the whole distance computation pipeline
    '''
    # Set up database connection
    connection = get_db_connection()

    # Get the cod id of all materials
    material_ids = get_all_material_ids(connection) #[0:100]

    # Interpolate all dos with respect to the material's hvb energy
    all_dos = interpolate_all_dos(connection, material_ids)

    # Remove this line if you want to compute all possible distances
    # material_ids = [1008776, 1008775, 1008787]
    material_ids = [1008776, 1008775, 1008787, 4060666, 4021827, 8000075]
    

    # Compute all the distances
    similarities_dict = {}
    k = 0
    for material_id in material_ids:
        # Start timer
        now = time.time()

        # Compute distances and sort
        try:
            similarities = find_similar_materials(connection, all_dos, material_id)
            similarities_sorted = sorted(similarities, key = lambda tup: tup[1])
            similarities_dict[material_id] = similarities_sorted[0:10]
        except Exception as e:
            later = time.time()
            difference = round(later - now, 3)
            print("Exception for material " + str(k+1) + "/" 
            + str(len(material_ids)) + " (COD ID: " + str(material_id) + ") in " 
            + str(difference) + " second(s)")
            continue

        # Display computation time
        later = time.time()
        difference = round(later - now, 3)
        print("Finished computation for material " + str(k+1) + "/" 
        + str(len(material_ids)) + " (COD ID: " + str(material_id) + ") in " 
        + str(difference) + " second(s)")
        k = k + 1

    # Persist the similarities dictionnary
    with open(OUTPUT_FILE, 'w+') as f:
        f.write(json.dumps(similarities_dict))

    # Close database connection
    connection.close()


def plot_similar_materials_3d(material_id):
    '''
    Plots top candidates for passed material.
    3D plots with a mosaic color style.
    '''
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Sets up the database connection
    connection = get_db_connection()

    # Finds all similar materials
    similarities_dict = {}
    with open(OUTPUT_FILE, 'r') as f:
        s = f.read()
        similarities_dict = ast.literal_eval(s)

    # Similarities for the current material
    current_similarities = similarities_dict[str(material_id)]

    # Set up plot dimensions
    xs = np.arange(0, 1000, 1.0)
    verts = []
    zs = np.arange(0, len(current_similarities), 1.0)
    max_values = []

    # Process each material
    for similarity in current_similarities:
        # Interpolate then plot the DOS
        dos = get_interpolated_dos(connection, similarity[0])
        ys = dos
        max_values.append(max(ys))
        ys[0], ys[-1] = 0, 0
        verts.append(list(zip(xs, ys)))

    # The mosaic color style
    colors = [(0.18, 0.5, 0.72, 1)] + [(np.random.rand(1)[0],
            np.random.rand(1)[0],np.random.rand(1)[0], 1-i) 
            for i in np.linspace(0.5, 1, len(verts)-1)]

    # Legends and colors
    poly = PolyCollection(verts, facecolors=colors)
    poly.set_alpha(0.7)
    ax.add_collection3d(poly, zs=zs, zdir='y')

    # Legends and colors
    ax.set_xlabel('X')
    ax.set_xlim3d(0, len(xs))
    ax.set_ylabel('Y')
    ax.set_ylim3d(-1, len(current_similarities))
    ax.set_zlabel('Z')
    ax.set_zlim3d(0, max(max_values))

    # Finally display the plots
    plt.show()


def plot_similar_materials(material_id):
    '''
    High-level method to display the DOS of some material and the similar ones
    '''
    # Set up database connection
    connection = get_db_connection()

    # Open the output of compute_distances() and read the dictionnary
    similarities_dict = {}
    with open(OUTPUT_FILE, 'r') as f:
        s = f.read()
        similarities_dict = ast.literal_eval(s)

    # Find the desired material and its similarities
    current_similarities = similarities_dict[str(material_id)]

    # Set the plot appearance
    from cycler import cycler
    colormap = plt.cm.hot
    plt.gca().set_prop_cycle(
        # First line is in the OMDB blue, next ones are gray with decreasing opacity
        cycler('color', [(0.18, 0.5, 0.72, 1)] 
        + [(0.4,0.4,0.4, 1-i) for i in np.linspace(0.5, 1, len(current_similarities)-1)]) +
        # Decreasing line width
        cycler('linewidth', [1-i for i in np.linspace(0, 0.5, len(current_similarities))])
    )

    # Get the similar materials and show them
    labels = []
    for similarity in current_similarities:
        # Interpolate then plot the DOS
        dos = get_interpolated_dos(connection, similarity[0])
        labels.append(similarity[0])
        print(dos)
        plt.plot(dos)
    plt.legend(labels)
    plt.show()


def export_similarities():
    '''
    Exports the top 10 similarities for each material to a database table.
    If the table does not exist in the omdb database, we will attempt
    to create it.
    '''
    # Set up database connection
    connection = get_db_connection()

    # Get the cod id of all materials
    material_ids = get_all_material_ids(connection) #[0:100]

    # Fetch all similarities
    similarities_dict = {}
    with open(OUTPUT_FILE, 'r') as f:
        s = f.read()
        similarities_dict = ast.literal_eval(s)

    # Iterate through all materials
    for material_id in material_ids:
        try:
            current_similarities = similarities_dict[str(material_id)]
        except Exception as e:
            #print("Probable KeyError on material " + str(material_id))
            continue
        try:
            with connection.cursor() as cursor:

                exists = False
                try:
                    # Read a single record
                    sql = (
                        'SELECT EXISTS(select * from similarities where reference=' + str(material_id) + ') '
                    )
                    # Parse result and return
                    cursor.execute(sql)
                    results = cursor.fetchone()
                    exists = True if (int(str(results)[-2:][:1]) == 1) else False
                    print("Exists " + str(material_id) + ": " + str(exists))
                except Exception as e:
                    print("Similarities table did not exist, so we will attempt to create it.")
                    create_similarities_table()
                    return

                sql = ""
                
                # Read a single record
                if (not exists):
                    sql = (
                        'INSERT INTO similarities '
                        '(reference, euc_1, euc_2, euc_3, euc_4, euc_5, '
                        'euc_6, euc_7, euc_8, euc_9, euc_10) '
                        'VALUES (' + str(material_id) + ', ' + str(current_similarities[0][0]) + ', ' 
                        + str(current_similarities[1][0]) + ', ' + str(current_similarities[2][0]) + ', ' 
                        + str(current_similarities[3][0]) + ', ' + str(current_similarities[4][0]) + ', ' 
                        + str(current_similarities[5][0]) + ', ' + str(current_similarities[6][0]) + ', ' 
                        + str(current_similarities[7][0]) + ', ' + str(current_similarities[8][0]) + ', ' 
                        + str(current_similarities[9][0]) + ') '
                    )
                else:
                    sql = (
                        'UPDATE similarities '
                        'SET euc_1=' + str(current_similarities[0][0]) + ', euc_2=' + 
                        str(current_similarities[1][0]) + ', euc_3=' + 
                        str(current_similarities[2][0]) + ', euc_4=' + 
                        str(current_similarities[3][0]) + ', euc_5=' + 
                        str(current_similarities[4][0]) + ', euc_6=' + 
                        str(current_similarities[5][0]) + ', euc_7=' + 
                        str(current_similarities[6][0]) + ', euc_8=' + 
                        str(current_similarities[7][0]) + ', euc_9=' + 
                        str(current_similarities[8][0]) + ', euc_10=' + 
                        str(current_similarities[9][0]) + ' '
                        'WHERE reference=' + str(material_id) + ' '
                    )

                # Parse result and return
                cursor.execute(sql)
                connection.commit()
                print(str(sql))
            with connection.cursor() as cursor:
                # Read a single record
                sql = (
                    'UPDATE similarities '
                    'SET euc_1_val=' + str(current_similarities[0][1]) + ', euc_2_val=' + 
                    str(current_similarities[1][1]) + ', euc_3_val=' + 
                    str(current_similarities[2][1]) + ', euc_4_val=' + 
                    str(current_similarities[3][1]) + ', euc_5_val=' + 
                    str(current_similarities[4][1]) + ', euc_6_val=' + 
                    str(current_similarities[5][1]) + ', euc_7_val=' + 
                    str(current_similarities[6][1]) + ', euc_8_val=' + 
                    str(current_similarities[7][1]) + ', euc_9_val=' + 
                    str(current_similarities[8][1]) + ', euc_10_val=' + 
                    str(current_similarities[9][1]) + ' '
                    'WHERE reference=' + str(material_id) + ' '
                )
                # Parse result and return
                cursor.execute(sql)
                connection.commit()
                print(str(sql))
                
        # If the similarities table is non-present we will end up here.
        except Exception as e:
            print("SQL Exception somewhere.")
            return
    
    print("Similarity export complete.")


def create_similarities_table():
    '''
    Creates a table used for storing top similarities
    based on each material's COD id.
    '''

    # Set up database connection
    connection = get_db_connection()

    # Connect and create the table
    with connection.cursor() as cursor:
        sql = (
            'CREATE TABLE similarities ( '
            'reference int, '
            'euc_1 int, '
            'euc_2 int, '
            'euc_3 int, '
            'euc_4 int, '
            'euc_5 int, '
            'euc_6 int, '
            'euc_7 int, '
            'euc_8 int, '
            'euc_9 int, '
            'euc_10 int, '
            'euc_1_val double, ' 
            'euc_2_val double, ' 
            'euc_3_val double, ' 
            'euc_4_val double, ' 
            'euc_5_val double, ' 
            'euc_6_val double, ' 
            'euc_7_val double, ' 
            'euc_8_val double, ' 
            'euc_9_val double, ' 
            'euc_10_val double ' 
            ' )'
        )
        cursor.execute(sql)
        connection.commit()
        print(str(sql))

    # Print something and return to the export function
    print("Successfully created similarities table.")
    export_similarities()


if __name__ == '__main__':
    # To run the distance computation:
    compute_distances()

    # To visualize results with plots:
    #plot_similar_materials(4060666)
    #plot_similar_materials_3d(4060666)

    # To export the materials to the similarities table:
    export_similarities()
