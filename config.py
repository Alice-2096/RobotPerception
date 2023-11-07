import yaml

"""
update the orb-slam2 configuration .yaml file with K matrix  
"""

def update_config(path, K): 

    with open(path, 'r') as yaml_file:
        data = yaml.safe_load(yaml_file)
        data['Camera.fx'] = float(K[0][0])
        data['Camera.fy'] = float(K[1][1])
        data['Camera.cx'] = float(K[0][2])
        data['Camera.cy'] = float(K[1][2])

    # write back to the file 
    with open(path, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)


