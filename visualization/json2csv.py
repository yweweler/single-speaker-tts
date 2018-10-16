import sys
import json
import numpy as np
import csv
import os


if __name__ == '__main__':
    """
    This tool converts scalar `.json` files exported from `tensorboard` into `.csv` files.
    The csv columns are written as ('time', 'step', 'value').
    
    Usage:
        python json2csv.py <in>.json <out>.csv
    """

    if len(sys.argv) != 3:
        print('Usage:')
        print('  python json2csv.py <in>.json <out>.csv')
        sys.exit(1)

    json_path = sys.argv[1]
    csv_path = sys.argv[2]

    if not os.path.exists(json_path):
        print('JSON file does not exist! ("{}")'.format(json_path))
        sys.exit(1)

    # Load data from JSON file.
    json_file = open(json_path, 'r')
    json_data = np.array(
        json.loads(json_file.read())
    )

    print('Writing to "{}" ...'.format(csv_path))
    # Open CSV file.
    csv_file = open(csv_path, 'w')
    csv_writer = csv.writer(csv_file)

    # Write CSV header.
    csv_writer.writerow(['time', 'step', 'value'])

    # Write CSV rows.
    for row in json_data:
        csv_writer.writerow(row)

    # Close CSV file.
    csv_file.close()
    print('DONE')